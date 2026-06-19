from __future__ import annotations
import numpy as np
from brahma_ca.rules.base import CARule
from brahma_ca.state import CAState


class AnomalyDetectionRule(CARule):
    """
    Detects cells whose energy significantly exceeds local + global background.
    Uses scipy.ndimage.uniform_filter for a vectorized sliding window mean.
    """

    def __init__(
        self,
        local_frame_radius: int = 43,   # ±43 frames ≈ ±250ms at hop=256/44kHz
        local_bin_radius: int = 10,     # ±10 bins ≈ ±430 Hz at fft=1024
        global_weight: float = 0.3,
        min_sigma: float = 2.0,
        write_anomaly: bool = True,
    ):
        self.local_frame_radius = local_frame_radius
        self.local_bin_radius = local_bin_radius
        self.global_weight = global_weight
        self.min_sigma = min_sigma
        self.write_anomaly = write_anomaly

    def apply(self, ca: CAState) -> CAState:
        from scipy.ndimage import uniform_filter

        frame_win = 2 * self.local_frame_radius + 1
        bin_win = 2 * self.local_bin_radius + 1
        local_bg = uniform_filter(ca.grid, size=(frame_win, bin_win), mode="reflect")

        global_mean = ca.grid.mean()
        global_std = ca.grid.std() + 1e-8

        bg = (1.0 - self.global_weight) * local_bg + self.global_weight * global_mean
        anomaly = np.clip((ca.grid - bg) / global_std, 0.0, None)

        if self.write_anomaly:
            ca.anomaly = anomaly

        amax = anomaly.max() + 1e-8
        ca.state = np.where(anomaly >= self.min_sigma, anomaly / amax, 0.0)
        return ca


class LocalOutlierRule(CARule):
    """
    Marks cells that exceed their local median by more than k * local MAD.
    More robust to impulsive noise than z-score.
    """

    def __init__(
        self,
        frame_radius: int = 21,
        bin_radius: int = 5,
        k: float = 3.0,
        write_anomaly: bool = True,
    ):
        self.frame_radius = frame_radius
        self.bin_radius = bin_radius
        self.k = k
        self.write_anomaly = write_anomaly

    def apply(self, ca: CAState) -> CAState:
        from scipy.ndimage import uniform_filter, median_filter

        fw = 2 * self.frame_radius + 1
        bw = 2 * self.bin_radius + 1

        local_med = median_filter(ca.grid, size=(fw, bw), mode="reflect")
        abs_dev = np.abs(ca.grid - local_med)
        local_mad = median_filter(abs_dev, size=(fw, bw), mode="reflect") + 1e-8

        score = (ca.grid - local_med) / local_mad
        score = np.clip(score, 0.0, None)

        if self.write_anomaly:
            ca.anomaly = score

        ca.state = np.where(score >= self.k, score / (score.max() + 1e-8), 0.0)
        return ca


class EdgeGatedAnomalyRule(CARule):
    """
    Composite: flags only anomalous cells that are also on spectral/temporal edges.
    Reduces false positives from sustained tonal content.
    """

    def __init__(
        self,
        edge_threshold_pct: float = 85.0,
        anomaly_min_sigma: float = 1.5,
    ):
        from brahma_ca.rules.edge import EdgeDetectionRule
        self._edge = EdgeDetectionRule(threshold_pct=edge_threshold_pct, write_mask=True)
        self._anom = AnomalyDetectionRule(min_sigma=anomaly_min_sigma, write_anomaly=True)

    def apply(self, ca: CAState) -> CAState:
        ca = self._edge.apply(ca)
        edge_mask = ca.edge_mask.copy()
        ca = self._anom.apply(ca)
        ca.state = ca.state * edge_mask.astype(np.float64)
        return ca
