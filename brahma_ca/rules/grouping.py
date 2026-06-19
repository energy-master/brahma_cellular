from __future__ import annotations
import numpy as np
from brahma_ca.rules.base import CARule
from brahma_ca.state import CAState


class GroupingRule(CARule):
    """
    Clusters active cells into connected labeled regions using scipy.ndimage.label.
    Filters small or thin regions via fully vectorized numpy operations.
    """

    def __init__(
        self,
        connectivity: int = 8,          # 4 = von Neumann, 8 = Moore neighborhood
        min_cells: int = 5,
        min_frame_span: int = 2,
        min_bin_span: int = 1,
        active_threshold: float = 0.1,
        write_labels: bool = True,
    ):
        self.connectivity = connectivity
        self.min_cells = min_cells
        self.min_frame_span = min_frame_span
        self.min_bin_span = min_bin_span
        self.active_threshold = active_threshold
        self.write_labels = write_labels

    def apply(self, ca: CAState) -> CAState:
        from scipy.ndimage import label

        binary = (ca.state > self.active_threshold).astype(np.int32)

        if self.connectivity == 8:
            structure = np.ones((3, 3), dtype=np.int32)
        else:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)

        labeled, n_regions = label(binary, structure=structure)

        if n_regions > 0:
            flat = labeled.ravel()
            n = n_regions + 1

            # Filter by cell count (vectorized)
            counts = np.bincount(flat, minlength=n)
            keep = counts >= self.min_cells
            keep[0] = False

            # Filter by frame span (vectorized using ufunc scatter)
            rows = np.repeat(np.arange(ca.frames, dtype=np.float64), ca.bins)
            min_f = np.full(n, float(ca.frames))
            max_f = np.zeros(n)
            np.minimum.at(min_f, flat, rows)
            np.maximum.at(max_f, flat, rows)
            keep &= (max_f - min_f + 1) >= self.min_frame_span

            # Filter by bin span (vectorized)
            cols = np.tile(np.arange(ca.bins, dtype=np.float64), ca.frames)
            min_b = np.full(n, float(ca.bins))
            max_b = np.zeros(n)
            np.minimum.at(min_b, flat, cols)
            np.maximum.at(max_b, flat, cols)
            keep &= (max_b - min_b + 1) >= self.min_bin_span

            # Zero out rejected regions
            labeled = np.where(keep[labeled], labeled, 0)

        if self.write_labels:
            ca.labels = labeled.astype(np.int32)

        ca.state = np.where(labeled > 0, ca.state, 0.0)
        return ca


class WatershedGroupingRule(CARule):
    """
    Seed from local maxima in ca.state and grow outward to adjacent active cells.
    Produces tighter region boundaries than connected-components when blobs overlap.
    """

    def __init__(
        self,
        min_peak_pct: float = 95.0,    # percentile threshold for seed peaks
        grow_steps: int = 5,
        active_threshold: float = 0.05,
        write_labels: bool = True,
    ):
        self.min_peak_pct = min_peak_pct
        self.grow_steps = grow_steps
        self.active_threshold = active_threshold
        self.write_labels = write_labels

    def apply(self, ca: CAState) -> CAState:
        from scipy.ndimage import label, maximum_filter

        # Find local maxima as seeds
        peak_thresh = np.percentile(ca.state, self.min_peak_pct)
        local_max = maximum_filter(ca.state, size=(3, 3)) == ca.state
        seeds = local_max & (ca.state >= peak_thresh)

        labeled, _ = label(seeds)

        active = ca.state > self.active_threshold

        # Iterative dilation confined to active region
        for _ in range(self.grow_steps):
            from scipy.ndimage import grey_dilation
            grown = grey_dilation(labeled, size=(3, 3)).astype(np.int32)
            labeled = np.where(active & (labeled == 0), grown, labeled)

        if self.write_labels:
            ca.labels = labeled.astype(np.int32)

        ca.state = np.where(labeled > 0, ca.state, 0.0)
        return ca
