from __future__ import annotations
import numpy as np
from brahma_cellular.rules.base import CARule
from brahma_cellular.state import CAState


class EdgeDetectionRule(CARule):
    """
    Detects temporal onsets/offsets and spectral transitions via gradient magnitude.
    Operates on ca.grid (the log-compressed spectrogram), not ca.state,
    so it can be the first rule in a chain.
    """

    def __init__(
        self,
        temporal_weight: float = 1.0,
        spectral_weight: float = 0.5,
        threshold_pct: float = 90.0,
        onset_only: bool = False,
        write_mask: bool = True,
    ):
        self.temporal_weight = temporal_weight
        self.spectral_weight = spectral_weight
        self.threshold_pct = threshold_pct
        self.onset_only = onset_only
        self.write_mask = write_mask

    def apply(self, ca: CAState) -> CAState:
        dt = ca.grid - np.roll(ca.grid, 1, axis=0)
        dt[0, :] = 0.0

        df = ca.grid - np.roll(ca.grid, 1, axis=1)
        df[:, 0] = 0.0

        grad = self.temporal_weight * np.abs(dt) + self.spectral_weight * np.abs(df)

        thresh = np.percentile(grad, self.threshold_pct)
        edge_mask = grad > thresh

        if self.onset_only:
            edge_mask &= dt > 0

        if self.write_mask:
            ca.edge_mask = edge_mask

        ca.state = np.where(edge_mask, grad / (grad.max() + 1e-8), 0.0)
        return ca


class SpectralFluxRule(CARule):
    """
    Per-bin energy difference between consecutive frames.
    Positive flux = onset; negative = offset.
    """

    def __init__(
        self,
        rectify: bool = True,       # keep only positive flux (onsets)
        threshold_pct: float = 85.0,
        write_mask: bool = True,
    ):
        self.rectify = rectify
        self.threshold_pct = threshold_pct
        self.write_mask = write_mask

    def apply(self, ca: CAState) -> CAState:
        flux = np.diff(ca.grid, axis=0, prepend=ca.grid[[0]])
        if self.rectify:
            flux = np.clip(flux, 0, None)

        thresh = np.percentile(flux, self.threshold_pct)
        mask = flux > thresh

        if self.write_mask:
            ca.edge_mask = mask

        ca.state = np.where(mask, flux / (flux.max() + 1e-8), 0.0)
        return ca
