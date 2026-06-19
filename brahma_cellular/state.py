from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CAState:
    grid:        np.ndarray        # (frames, bins) float64 — log1p-compressed spectrogram
    state:       np.ndarray        # (frames, bins) float64 — current CA activation [0, 1]
    fft_size:    int = 1024
    hop_size:    int = 256
    sample_rate: int = 44100
    edge_mask:   np.ndarray | None = None   # (frames, bins) bool
    anomaly:     np.ndarray | None = None   # (frames, bins) float64
    labels:      np.ndarray | None = None   # (frames, bins) int32 region IDs
    step:        int = 0
    history:     list[np.ndarray] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return self.grid.shape[0]

    @property
    def bins(self) -> int:
        return self.grid.shape[1]

    @property
    def bin_hz(self) -> np.ndarray:
        return np.arange(self.bins) * (self.sample_rate / self.fft_size)

    @property
    def frame_sec(self) -> np.ndarray:
        return np.arange(self.frames) * (self.hop_size / self.sample_rate)

    @classmethod
    def from_fourier(cls, fourier: dict, log_compress: bool = True) -> "CAState":
        mags = np.asarray(fourier["magnitudes"], dtype=np.float64)
        frames = fourier["frames"]
        bins = fourier["bins"]
        grid = mags.reshape(frames, bins)
        if log_compress:
            grid = np.log1p(grid)
        stft_p = fourier.get("stft", {})
        return cls(
            grid=grid,
            state=np.zeros((frames, bins), dtype=np.float64),
            fft_size=stft_p.get("fft", 1024),
            hop_size=stft_p.get("hop", 256),
            sample_rate=stft_p.get("sample_rate", fourier.get("sample_rate", 44100)),
        )

    @classmethod
    def from_signal(cls, signal: np.ndarray, sample_rate: int = 44100,
                    fft_size: int = 1024, hop_size: int | None = None,
                    window: str = "hann", log_compress: bool = True) -> "CAState":
        from identdynamics import stft  # type: ignore[import-untyped]
        hop = hop_size if hop_size is not None else fft_size // 4
        magnitudes, frames, bins = stft(signal, fft_size=fft_size, hop_size=hop, window=window)
        fourier = {
            "magnitudes": magnitudes,
            "frames": frames,
            "bins": bins,
            "sample_rate": sample_rate,
            "stft": {"fft": fft_size, "hop": hop, "window": window, "sample_rate": sample_rate},
        }
        return cls.from_fourier(fourier, log_compress=log_compress)
