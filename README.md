# brahma_cellular

Cellular automata rules for acoustic signal detection, edge finding, and signature labelling — built on NumPy and SciPy, designed to run against spectrograms from the [identdynamics](https://github.com/energy-master/identdynamics-sdk) SDK.

## Install

```bash
# From GitHub
pip install "brahma_cellular[identdynamics] @ git+https://github.com/energy-master/brahma_cellular.git"

# Without identdynamics (bring your own STFT)
pip install "brahma_cellular @ git+https://github.com/energy-master/brahma_cellular.git"
```

## Quick start

```python
from identdynamics import Client
from brahma_cellular import (
    Pipeline,
    EdgeDetectionRule, AnomalyDetectionRule, GroupingRule, WolframRule,
)

client = Client(base_url="https://goident.ai", token="<your-token>")
fourier = client.fourier_for_file(file_id)

# Build a rule chain with | syntax
rule = WolframRule(rule_number=90, steps=3) | AnomalyDetectionRule(min_sigma=1.5) | GroupingRule()
pipe = Pipeline(rule=rule, threshold=0.45)

result = pipe.run(fourier)
# result["scores"]     — float list, one value per spectrogram frame
# result["detections"] — list of {start, end, start_sec, end_sec, label, score}

# Post back to identdynamics
client.post_run(target={"kind": "file", "id": file_id}, **{k: v for k, v in result.items() if k != "_ca"})
```

## Rules

| Class | Description |
|---|---|
| `EdgeDetectionRule` | Gradient magnitude across time and frequency — finds onsets, offsets, spectral transitions |
| `SpectralFluxRule` | Per-bin inter-frame energy difference — fast onset detector |
| `AnomalyDetectionRule` | Local z-score vs sliding window background (scipy `uniform_filter`) |
| `LocalOutlierRule` | Median Absolute Deviation — robust to impulsive noise |
| `EdgeGatedAnomalyRule` | Composite: anomaly cells that are also on edges |
| `GroupingRule` | Connected-component labelling with size/span filters |
| `WatershedGroupingRule` | Seed from peaks, grow outward into active regions |
| `WolframRule` | 1D Wolfram elementary CA (rules 0–255) applied along frequency axis per frame |
| `TotalisticRule` | 2D Game-of-Life variant with tunable birth/survive ranges |

Rules compose via `|`:

```python
rule = EdgeDetectionRule(onset_only=True) | GroupingRule(min_cells=10)
```

## Working with raw audio (no identdynamics)

```python
import numpy as np
from brahma_cellular import CAState, CAEngine, Detector, EdgeDetectionRule

signal = np.frombuffer(open("clip.raw", "rb").read(), dtype=np.float32).astype(np.float64)
ca = CAState.from_signal(signal, sample_rate=44100)   # requires identdynamics for STFT

# Or build from any 2D spectrogram:
spectrogram = np.random.rand(200, 513)   # (frames, bins)
fourier = {"magnitudes": spectrogram.ravel(), "frames": 200, "bins": 513,
           "sample_rate": 44100, "stft": {"fft": 1024, "hop": 256}}
ca = CAState.from_fourier(fourier)

engine = CAEngine(rule=EdgeDetectionRule(), steps=1)
ca = engine.evolve(ca)

detector = Detector(threshold=0.4)
print(detector.detections(ca))
```

## Notable Wolfram rules for acoustics

| Rule | Character |
|---|---|
| 30  | Chaotic — noise floor / broadband event detector |
| 90  | XOR — emphasises harmonic periodicity |
| 110 | Complex — sustained pattern evolution |
| 150 | XOR+centre — amplifies isolated spectral peaks |

## Frequency band labels

`Labeler` assigns names from the default band map to detections and regions:

`sub_bass · bass · low_mid · mid · upper_mid · presence · brilliance`

Custom bands:

```python
from brahma_cellular import Labeler
labeler = Labeler(bands={"engine": (80, 300), "propeller": (300, 1200)})
dets = labeler.label_detections(ca, detections)
```

## License

MIT
