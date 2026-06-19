from __future__ import annotations
import numpy as np
from brahma_cellular.state import CAState


class Detector:
    """
    Converts a CA state (frames × bins) into a per-frame score vector
    and a list of detection dicts compatible with identdynamics post_run().
    """

    def __init__(
        self,
        score_mode: str = "max",        # 'max', 'mean', 'sum', 'density'
        threshold: float = 0.5,
        min_gap_frames: int = 5,        # merge detections separated by < N frames
        min_dur_frames: int = 3,        # discard events shorter than N frames
        normalize: bool = True,
    ):
        if score_mode not in ("max", "mean", "sum", "density"):
            raise ValueError(f"score_mode must be one of max/mean/sum/density, got {score_mode!r}")
        self.score_mode = score_mode
        self.threshold = threshold
        self.min_gap_frames = min_gap_frames
        self.min_dur_frames = min_dur_frames
        self.normalize = normalize

    def scores(self, ca: CAState) -> np.ndarray:
        if self.score_mode == "max":
            s = ca.state.max(axis=1)
        elif self.score_mode == "mean":
            s = ca.state.mean(axis=1)
        elif self.score_mode == "sum":
            s = ca.state.sum(axis=1)
        else:  # density
            s = (ca.state > 0).sum(axis=1).astype(np.float64) / ca.bins

        if self.normalize:
            s = s / (s.max() + 1e-8)

        return s.astype(np.float64)

    def detections(self, ca: CAState, label: str = "event") -> list[dict]:
        s = self.scores(ca)
        active = (s > self.threshold).astype(np.int8)

        diff = np.diff(active, prepend=np.int8(0), append=np.int8(0))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0:
            return []

        # Merge gaps shorter than min_gap_frames
        if len(starts) > 1:
            gaps = starts[1:] - ends[:-1]
            merge = gaps < self.min_gap_frames
            keep_start = np.concatenate([[True], ~merge])
            keep_end = np.concatenate([~merge, [True]])
            starts = starts[keep_start]
            ends = ends[keep_end]

        # Filter short events
        durations = ends - starts
        valid = durations >= self.min_dur_frames
        starts = starts[valid]
        ends = ends[valid]

        frame_sec = ca.frame_sec
        return [
            {
                "start": int(st),
                "end": int(en),
                "start_sec": float(frame_sec[st]),
                "end_sec": float(frame_sec[min(en, ca.frames - 1)]),
                "label": label,
                "score": float(s[st:en].max()),
            }
            for st, en in zip(starts, ends)
        ]
