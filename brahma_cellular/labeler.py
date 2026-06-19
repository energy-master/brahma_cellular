from __future__ import annotations
import numpy as np
from brahma_cellular.state import CAState

FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "sub_bass":   (0.0,    60.0),
    "bass":       (60.0,   250.0),
    "low_mid":    (250.0,  500.0),
    "mid":        (500.0,  2000.0),
    "upper_mid":  (2000.0, 4000.0),
    "presence":   (4000.0, 6000.0),
    "brilliance": (6000.0, 22050.0),
}


class Labeler:
    """
    Assigns frequency band label strings to detections and CA regions
    based on which bins hold the most active energy.
    """

    def __init__(
        self,
        bands: dict[str, tuple[float, float]] = FREQUENCY_BANDS,
        multi_label: bool = True,
        min_band_pct: float = 0.10,
        fallback_label: str = "event",
    ):
        self.bands = bands
        self.multi_label = multi_label
        self.min_band_pct = min_band_pct
        self.fallback_label = fallback_label

    def _dominant_bands(self, segment: np.ndarray, bin_hz: np.ndarray) -> str:
        total_active = float((segment > 0).sum()) + 1e-8
        chosen = []
        for name, (lo, hi) in self.bands.items():
            mask = (bin_hz >= lo) & (bin_hz < hi)
            band_active = float((segment[:, mask] > 0).sum())
            if band_active / total_active >= self.min_band_pct:
                chosen.append(name)
                if not self.multi_label:
                    break
        return "+".join(chosen) if chosen else self.fallback_label

    def label_detections(self, ca: CAState, detections: list[dict]) -> list[dict]:
        bin_hz = ca.bin_hz
        for det in detections:
            seg = ca.state[det["start"]:det["end"], :]
            det["label"] = self._dominant_bands(seg, bin_hz)
        return detections

    def label_regions(self, ca: CAState) -> dict[int, str]:
        if ca.labels is None:
            return {}

        bin_hz = ca.bin_hz
        region_ids = np.unique(ca.labels)
        region_ids = region_ids[region_ids > 0]

        result: dict[int, str] = {}
        for rid in region_ids:
            mask = ca.labels == rid          # (frames, bins) bool
            energy = ca.grid * mask          # zero out non-region cells

            # Energy-weighted dominant band
            band_scores: dict[str, float] = {}
            for name, (lo, hi) in self.bands.items():
                band_mask = (bin_hz >= lo) & (bin_hz < hi)
                band_scores[name] = float(energy[:, band_mask].sum())

            dominant = max(band_scores, key=band_scores.get)  # type: ignore[arg-type]
            result[int(rid)] = dominant

        return result
