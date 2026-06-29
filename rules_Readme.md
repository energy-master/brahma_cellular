# Brahma Cellular — Rules Reference

A field guide to every rule, neighbourhood type, and update strategy in the library.

---

## How the CA Algorithm Works

A cellular automaton treats a 2D grid of numbers as a population of cells. Each cell holds a value and, at each time step, that value is recomputed from the cell's own current value and the values of its neighbours. Apply that update simultaneously to every cell and you have one CA generation.

In Brahma Cellular the grid is a spectrogram: rows are time frames, columns are frequency bins, and each cell's value is the log-compressed magnitude at that (time, frequency) coordinate. The CA evolves over this grid to highlight structure — edges, anomalies, clusters — that is hard to see in the raw numbers.

### The CA matrix: initial values

When a `CAState` is built from audio (via `CAState.from_fourier` or `CAState.from_signal`), two arrays are created:

```
ca.grid   — shape (frames, bins), float64
            The log1p-compressed STFT magnitudes. This is read-only input;
            rules read from it but do not modify it.

ca.state  — shape (frames, bins), float64, initialised to all zeros
            The CA activation layer. Rules write their output here.
            Values are normalised to [0, 1].
```

Before the first rule runs, `CAEngine` optionally binarises the grid into `ca.state`:

```python
threshold = percentile(ca.grid, threshold_pct)   # default: 80th percentile
ca.state  = (ca.grid > threshold).astype(float)  # 1.0 where loud, 0.0 elsewhere
```

This gives the first rule a clean binary "active / inactive" map to work with. Rules that read directly from `ca.grid` (like `EdgeDetectionRule`) skip this and see the full continuous magnitudes instead.

### One update step — worked example with SpectralFluxRule

Suppose `ca.grid` is a tiny 4-frame × 4-bin spectrogram (log magnitudes):

```
         bin0  bin1  bin2  bin3
frame 0 [ 0.1   0.2   0.1   0.0 ]
frame 1 [ 0.1   0.8   0.1   0.0 ]   ← energy spike in bin1
frame 2 [ 0.1   0.7   0.5   0.0 ]
frame 3 [ 0.1   0.3   0.2   0.0 ]
```

**Step 1 — compute flux** (difference from previous frame, per bin):

```
         bin0  bin1  bin2  bin3
frame 0 [ 0.0   0.0   0.0   0.0 ]   (first frame: no previous, set to 0)
frame 1 [ 0.0  +0.6   0.0   0.0 ]   ← onset: bin1 jumped by 0.6
frame 2 [ 0.0  -0.1  +0.4   0.0 ]   ← bin2 rose, bin1 fell
frame 3 [ 0.0  -0.4  -0.3   0.0 ]   ← both falling
```

**Step 2 — rectify** (clip negatives to 0, keeping only onsets):

```
frame 1 [ 0.0  +0.6   0.0   0.0 ]
frame 2 [ 0.0   0.0  +0.4   0.0 ]
frame 3 [ 0.0   0.0   0.0   0.0 ]
```

**Step 3 — threshold** at 85th percentile of the rectified values. Only the two non-zero cells survive; the rest become 0.

**Step 4 — normalise** by dividing by the maximum value (0.6):

```
ca.state after update:
         bin0  bin1  bin2  bin3
frame 0 [ 0.0   0.0   0.0   0.0 ]
frame 1 [ 0.0   1.0   0.0   0.0 ]   ← strong onset marked
frame 2 [ 0.0   0.0   0.67  0.0 ]   ← weaker onset marked
frame 3 [ 0.0   0.0   0.0   0.0 ]
```

Every cell has been updated in one vectorised pass — no Python loop over cells. `ca.step` increments by 1. If `steps > 1`, this updated `ca.state` feeds directly into the next iteration.

### Chaining rules

When rules are chained with `|`, each rule's output `ca.state` becomes the next rule's input. Auxiliary arrays (`ca.edge_mask`, `ca.anomaly`, `ca.labels`) are written alongside `ca.state` so that later rules can gate on earlier results:

```
audio signal
    → ca.grid  (fixed, read-only log spectrogram)
    → ca.state (zeros)
        → EdgeDetectionRule  writes ca.state + ca.edge_mask
        → AnomalyDetectionRule  reads ca.grid, writes ca.state gated by ca.edge_mask
        → GroupingRule  reads ca.state, writes ca.labels
        → Detector  reads ca.state + ca.labels → events with timestamps
```

---

## Neighbourhood Types

| Name | Shape | Used By | Description |
|---|---|---|---|
| **1D Linear** | 1 × 3 | WolframRule | `[left, centre, right]` along the frequency (bin) axis |
| **3×3 Moore** | 3 × 3 | TotalisticRule, GroupingRule (8-conn), WatershedGroupingRule | All 8 orthogonal + diagonal neighbours |
| **3×3 von Neumann** | 3 × 3 | GroupingRule (4-conn) | Orthogonal neighbours only (up/down/left/right) |
| **2D Sliding Window** | (2r+1) × (2c+1) | AnomalyDetectionRule, LocalOutlierRule | Arbitrary radius in frames and bins; boundary mode = `reflect` |

---

## Rules

### EdgeDetectionRule

**File:** `brahma_cellular/rules/edge.py`

Detects temporal onsets/offsets and spectral transitions by computing gradient magnitude across the log-compressed spectrogram (`ca.grid`). Because it operates on `ca.grid` rather than `ca.state`, it can be placed first in any chain without requiring prior activation.

| Parameter | Default | Meaning |
|---|---|---|
| `temporal_weight` | `1.0` | Weight applied to the time-axis gradient |
| `spectral_weight` | `0.5` | Weight applied to the frequency-axis gradient |
| `threshold_pct` | `90.0` | Percentile used to threshold the combined gradient magnitude |
| `onset_only` | `False` | When `True`, only rising-energy transitions (onsets) are kept |
| `write_mask` | `True` | Stores a binary edge mask in `ca.edge_mask` for downstream rules |

**Neighbourhood:** 2D gradient kernel (finite differences along each axis independently).

**How it finds edges in the spectrogram:**

An edge in the spectrogram is any place where magnitude changes sharply — either across time (an onset or offset) or across frequency (a spectral boundary between a tone and silence). The rule computes two finite-difference arrays from `ca.grid`:

```python
dt = ca.grid - roll(ca.grid, 1, axis=0)   # frame[t] − frame[t−1], per bin
df = ca.grid - roll(ca.grid, 1, axis=1)   # bin[b] − bin[b−1], per frame
```

`dt` is large wherever energy jumps or drops between consecutive frames — the signature of an onset or offset. `df` is large wherever energy jumps between adjacent frequency bins — the signature of a spectral boundary (e.g., the edge of a harmonic or a band of noise). Both are combined by weighted magnitude:

```python
grad = temporal_weight * |dt| + spectral_weight * |df|
```

Because both are absolute values, `grad` is high at any sharp change, regardless of direction. The result is thresholded at a percentile so only the most prominent transitions survive. Setting `onset_only=True` adds a sign check (`dt > 0`) so only rising energy is kept. The output written to `ca.state` is the normalised gradient magnitude — stronger edges have values closer to 1.

**Good for:** Fast onset/offset detection; feeding an edge mask into anomaly or grouping rules; pre-conditioning the state for any rule that benefits from knowing where energy changes sharply.

---

### SpectralFluxRule

**File:** `brahma_cellular/rules/edge.py`

Computes the per-bin energy difference between consecutive frames (`diff` along the frame axis). Positive flux = energy rising (onset); negative = energy falling (offset).

| Parameter | Default | Meaning |
|---|---|---|
| `rectify` | `True` | Clip negative flux to zero, keeping only onsets |
| `threshold_pct` | `85.0` | Percentile mask applied after rectification |
| `write_mask` | `True` | Stores binary mask in `ca.edge_mask` |

**Neighbourhood:** Single-frame look-back (1D along time axis).

**How it finds edges in the spectrogram:**

Spectral flux is the simplest possible edge detector: for each frequency bin independently, subtract the previous frame's magnitude from the current one:

```python
flux[t, b] = ca.grid[t, b] − ca.grid[t−1, b]
```

A positive value means energy in that bin increased this frame — an onset. A negative value means it decreased — an offset. With `rectify=True` the negative values are clipped to zero so only onsets remain. Unlike `EdgeDetectionRule`, which blends temporal and spectral gradients into a 2D magnitude, SpectralFluxRule looks purely along the time axis one bin at a time. This makes it more sensitive to sharp per-bin jumps (e.g., a single harmonic appearing suddenly) and less sensitive to smooth spectral slopes. The threshold percentile then discards the low-flux baseline, leaving only the bins and frames where energy changed most dramatically.

**Good for:** Onset detection when frame-by-frame energy jumps matter more than sustained gradient; faster and simpler than `EdgeDetectionRule`; best when audio is already fairly clean.

---

### AnomalyDetectionRule

**File:** `brahma_cellular/rules/anomaly.py`

Flags cells whose energy significantly exceeds a blended local + global background using a z-score. Local background is estimated by a 2D sliding window (uniform filter); global statistics are computed over the whole grid.

| Parameter | Default | Meaning |
|---|---|---|
| `local_frame_radius` | `43` | Half-width of the sliding window in frames (≈ ±250 ms at 44 kHz, hop 256) |
| `local_bin_radius` | `10` | Half-width of the sliding window in bins (≈ ±430 Hz at FFT 1024) |
| `global_weight` | `0.3` | Blend factor: `bg = (1 − gw)·local + gw·global_mean` |
| `min_sigma` | `2.0` | Minimum z-score to flag a cell as anomalous |
| `write_anomaly` | `True` | Stores anomaly scores in `ca.anomaly` |

**Neighbourhood:** 2D sliding window of size `(2·local_frame_radius+1) × (2·local_bin_radius+1)`.

**Good for:** Detecting unusual energy patches (unexpected tones, artifacts, non-stationary events) against a smoothly varying background; works well on music and speech where the local mean is informative.

---

### LocalOutlierRule

**File:** `brahma_cellular/rules/anomaly.py`

Marks cells that exceed their local median by more than `k × local MAD` (Median Absolute Deviation). MAD-based scoring is more robust to impulsive noise and heavy-tailed distributions than z-score.

| Parameter | Default | Meaning |
|---|---|---|
| `frame_radius` | `21` | Half-width of the median window in frames |
| `bin_radius` | `5` | Half-width of the median window in bins |
| `k` | `3.0` | Sensitivity: cells with score ≥ k are flagged |
| `write_anomaly` | `True` | Stores scores in `ca.anomaly` |

**Neighbourhood:** 2D sliding window via `scipy.ndimage.median_filter`.

**Good for:** Robust anomaly detection when noise is impulsive or the signal violates Gaussian assumptions; better than `AnomalyDetectionRule` on recordings with sharp transients or crackling.

---

### EdgeGatedAnomalyRule

**File:** `brahma_cellular/rules/anomaly.py`

Composite rule: flags only cells that are simultaneously on a spectral/temporal edge **and** anomalous. Internally applies `EdgeDetectionRule` then `AnomalyDetectionRule` and multiplies the results (logical AND).

| Parameter | Default | Meaning |
|---|---|---|
| `edge_threshold_pct` | `85.0` | Percentile for edge detection |
| `anomaly_min_sigma` | `1.5` | Z-score threshold for anomaly detection |

**Neighbourhood:** Inherits from both constituent rules (gradient kernel + 2D sliding window).

**Good for:** High-precision detection — reduces false positives from sustained tonal content by requiring that an anomaly must also occur at an energy transition; useful when you care more about precision than recall.

---

### GroupingRule

**File:** `brahma_cellular/rules/grouping.py`

Clusters active cells into connected labeled regions using `scipy.ndimage.label`, then filters small or thin regions via fully vectorised NumPy operations.

| Parameter | Default | Meaning |
|---|---|---|
| `connectivity` | `8` | `4` = von Neumann (ortho only), `8` = Moore (ortho + diagonal) |
| `min_cells` | `5` | Minimum active cell count per region |
| `min_frame_span` | `2` | Minimum number of frames spanned (temporal height) |
| `min_bin_span` | `1` | Minimum number of bins spanned (spectral width) |
| `active_threshold` | `0.1` | `ca.state > threshold` → binary mask for labeling |
| `write_labels` | `True` | Stores integer label map in `ca.labels` |

**Neighbourhood:** 4-connectivity uses von Neumann; 8-connectivity uses Moore.

**Good for:** Segmenting detection output into discrete events; filtering spurious single-cell activations or temporally thin detections; downstream event extraction and labeling.

---

### WatershedGroupingRule

**File:** `brahma_cellular/rules/grouping.py`

Seeds from local maxima in `ca.state` above a percentile threshold, then iteratively dilates labels outward into active cells. Produces tighter boundaries between overlapping blobs than connected-component labeling.

| Parameter | Default | Meaning |
|---|---|---|
| `min_peak_pct` | `95.0` | Percentile threshold for selecting seed peaks |
| `grow_steps` | `5` | Number of grey-dilation iterations |
| `active_threshold` | `0.05` | Cells with `ca.state > threshold` are growable |
| `write_labels` | `True` | Stores labeled map in `ca.labels` |

**Neighbourhood:** 3 × 3 Moore for both peak detection and dilation steps.

**Good for:** Separating overlapping or touching blobs more accurately than `GroupingRule`; cases where two events are temporally adjacent and connected-component labeling would merge them.

---

### WolframRule

**File:** `brahma_cellular/rules/wolfram.py`

Applies a 1D Wolfram elementary CA rule (0–255) along the frequency (bin) axis for all frames simultaneously — no Python loop over frames. Each cell's next state is determined by encoding its `[left, centre, right]` 3-cell neighbourhood as a 3-bit index into a rule lookup table.

| Parameter | Default | Meaning |
|---|---|---|
| `rule_number` | `30` | Wolfram rule 0–255 |
| `steps` | `1` | Number of CA generations to evolve |
| `binarize` | `True` | Threshold `ca.state` before applying the rule |
| `threshold_pct` | `50.0` | Percentile used when binarizing |
| `boundary` | `"wrap"` | `'wrap'` = periodic boundary; `'zero'` = zero-pad edges |

**Neighbourhood:** 1D Linear — `[left, centre, right]` along the bin axis.

**Notable rules for audio:**

| Rule | Character | Good for |
|---|---|---|
| 30 | Chaotic | Broadband / noise detection; distinguishes noise-like from tonal content |
| 90 | XOR | Harmonic emphasis; reinforces periodic spectral structure |
| 110 | Complex | Pattern evolution; Turing-complete, produces rich structure |
| 150 | XOR + centre | Peak amplification; isolates narrow spectral peaks |

**Good for:** Pattern evolution along the frequency axis; emphasising or suppressing harmonic structure; exploring how energy patterns propagate across bins over multiple generations.

---

### TotalisticRule

**File:** `brahma_cellular/rules/wolfram.py`

A 2D outer totalistic CA (Game-of-Life variant) with tunable birth and survival ranges. Each cell's next state depends on its current alive/dead status and the sum of its 3 × 3 Moore neighbourhood. Operates on continuous `[0, 1]` values; surviving or born cells inherit the original intensity.

| Parameter | Default | Meaning |
|---|---|---|
| `birth_range` | `(2.0, 3.5)` | Neighbourhood sum range that births a dead cell |
| `survive_range` | `(1.5, 4.5)` | Neighbourhood sum range that keeps a live cell alive |
| `active_threshold` | `0.3` | `ca.state > threshold` counts as "alive" |
| `steps` | `1` | Number of generations |

**Neighbourhood:** 3 × 3 Moore (8 neighbours + centre), reduced to a scalar sum.

**Good for:** Region stability testing — Life-like rules quickly kill isolated noise cells while preserving and growing spatially coherent patterns; tuning `birth_range`/`survive_range` lets you control how aggressively sparse activations are suppressed.

---

## Update Strategies

### RuleChain (composition via `|`)

**File:** `brahma_cellular/rules/base.py`

Rules can be composed sequentially with the `|` operator. Each rule's output `CAState` becomes the next rule's input.

```python
rule = EdgeDetectionRule() | AnomalyDetectionRule() | GroupingRule()
```

The chain is applied in left-to-right order. Intermediate results propagate through `ca.state` and auxiliary fields (`ca.edge_mask`, `ca.anomaly`, `ca.labels`).

---

### CAEngine (evolution over multiple steps)

**File:** `brahma_cellular/engine.py`

Orchestrates repeated rule application and optional history tracking.

| Parameter | Default | Meaning |
|---|---|---|
| `rule` | — | The rule or `RuleChain` to evolve |
| `steps` | `1` | Number of evolution iterations |
| `save_history` | `False` | Append `ca.state` to `ca.history` at each step |
| `binarize_input` | `True` | Pre-process grid via percentile threshold before first step |
| `threshold_pct` | `80.0` | Percentile for input binarization |

**Workflow:** optionally binarize → for each step: (optionally save history → apply rule → increment `ca.step`).

---

### Detector (per-frame scoring and event extraction)

**File:** `brahma_cellular/detector.py`

Converts the final CA state grid into per-frame scalar scores and a list of time-bounded events.

| Parameter | Default | Meaning |
|---|---|---|
| `score_mode` | `"max"` | Aggregation per frame: `max`, `mean`, `sum`, or `density` |
| `threshold` | `0.5` | Score threshold for event detection |
| `min_gap_frames` | `5` | Merge events separated by fewer than N frames |
| `min_dur_frames` | `3` | Discard events shorter than N frames |
| `normalize` | `True` | Normalize scores to `[0, 1]` |

**Score modes:**

| Mode | Formula | Best when |
|---|---|---|
| `max` | `max(state[frame, :])` | A single strong bin drives the detection |
| `mean` | `mean(state[frame, :])` | Energy is spread across many bins |
| `sum` | `sum(state[frame, :])` | Total activated mass matters |
| `density` | `count(>0) / num_bins` | Fraction of active bins is the signal |

---

### Pipeline (end-to-end orchestration)

**File:** `brahma_cellular/pipeline.py`

High-level entry point: STFT dict → CA evolution → detection → labeling → output dict.

| Parameter | Default | Meaning |
|---|---|---|
| `rule` | — | Rule or `RuleChain` |
| `model_name` | `"ca_pipeline"` | Key used in the output dict |
| `steps` | `1` | Evolution steps passed to `CAEngine` |
| `threshold` | `0.5` | Detection threshold for `Detector` |
| `score_mode` | `"max"` | Frame aggregation mode |
| `label` | `True` | Assign frequency band labels to detections |
| `normalize` | `True` | Normalize scores to `[0, 1]` |
| `binarize_input` | `True` | Pre-process grid before evolution |
| `threshold_pct` | `80.0` | Percentile for input binarization |
| `save_history` | `False` | Store per-step state history |

---

### Labeler (frequency band assignment)

**File:** `brahma_cellular/labeler.py`

Assigns frequency band labels to each detection based on which bands contain the most active cells.

**Default bands:**

| Label | Range |
|---|---|
| `sub_bass` | 0 – 60 Hz |
| `bass` | 60 – 250 Hz |
| `low_mid` | 250 – 500 Hz |
| `mid` | 500 – 2000 Hz |
| `upper_mid` | 2000 – 4000 Hz |
| `presence` | 4000 – 6000 Hz |
| `brilliance` | 6000 – 22050 Hz |

| Parameter | Default | Meaning |
|---|---|---|
| `multi_label` | `True` | Include all bands meeting the threshold, not just the top one |
| `min_band_pct` | `0.10` | Band must account for ≥ 10 % of active cells to be included |
| `fallback_label` | `"event"` | Used when no band meets the threshold |

---

## Common Recipe Chains

```python
# Fast onset detection with region segmentation
SpectralFluxRule(rectify=True) | GroupingRule(min_cells=5)

# High-precision anomaly (edge-gated) with connected-component cleanup
EdgeGatedAnomalyRule() | GroupingRule()

# Harmonic emphasis via Wolfram rule 90, then anomaly scoring
WolframRule(rule_number=90, steps=3) | AnomalyDetectionRule() | GroupingRule()

# Robust outlier detection with watershed clustering
LocalOutlierRule(k=2.5) | WatershedGroupingRule()

# Full pipeline: edge → anomaly → watershed → events with band labels
rule = EdgeDetectionRule() | AnomalyDetectionRule() | WatershedGroupingRule()
Pipeline(rule=rule, score_mode="density", label=True).run(fourier_dict)
```
