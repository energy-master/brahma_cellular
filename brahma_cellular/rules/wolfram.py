from __future__ import annotations
import numpy as np
from brahma_cellular.rules.base import CARule
from brahma_cellular.state import CAState

# Notable rules for acoustic analysis:
#   30  — chaotic, good noise / broadband event detector
#   90  — XOR (Pascal's triangle mod 2), emphasises harmonic periodicity
#   110 — complex / Turing-complete, sustained pattern evolution
#   150 — XOR with centre, amplifies isolated spectral peaks
NOTABLE_RULES: dict[int, str] = {
    30: "chaotic / noise detector",
    90: "XOR / harmonic emphasis",
    110: "complex / pattern evolution",
    150: "XOR+centre / peak amplification",
}


def _build_rule_table(rule_number: int) -> np.ndarray:
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)


class WolframRule(CARule):
    """
    Apply a 1D Wolfram elementary CA rule (0-255) along the frequency (bin) axis
    for all frames simultaneously — no Python loop over frames.

    Each step maps every (frame, bin) cell to a new value determined by the
    3-cell neighbourhood [left, centre, right] packed into a 3-bit integer
    that indexes the rule lookup table.
    """

    def __init__(
        self,
        rule_number: int = 30,
        steps: int = 1,
        binarize: bool = True,
        threshold_pct: float = 50.0,
        boundary: str = "wrap",     # 'wrap' or 'zero'
    ):
        if not 0 <= rule_number <= 255:
            raise ValueError(f"rule_number must be 0-255, got {rule_number}")
        self.rule_number = rule_number
        self.steps = steps
        self.binarize = binarize
        self.threshold_pct = threshold_pct
        self.boundary = boundary
        self._table = _build_rule_table(rule_number)

    def apply(self, ca: CAState) -> CAState:
        if self.binarize:
            pct = np.percentile(ca.state, self.threshold_pct)
            s = (ca.state > pct).astype(np.uint8)
        else:
            s = (ca.state > 0).astype(np.uint8)

        table = self._table

        for _ in range(self.steps):
            left = np.roll(s, 1, axis=1)
            right = np.roll(s, -1, axis=1)

            if self.boundary == "zero":
                left[:, 0] = 0
                right[:, -1] = 0

            idx = (left.astype(np.int32) << 2) | (s.astype(np.int32) << 1) | right.astype(np.int32)
            s = table[idx]

        ca.state = s.astype(np.float64)
        return ca


class TotalisticRule(CARule):
    """
    2D outer totalistic CA: next state depends on current cell value and
    the sum of its Moore neighbourhood. Operates on continuous [0,1] values.
    Uses scipy.ndimage.uniform_filter for the neighbourhood sum — fully vectorized.
    """

    def __init__(
        self,
        birth_range: tuple[float, float] = (2.0, 3.5),    # neighbourhood sum to birth
        survive_range: tuple[float, float] = (1.5, 4.5),  # neighbourhood sum to survive
        active_threshold: float = 0.3,
        steps: int = 1,
    ):
        self.birth_range = birth_range
        self.survive_range = survive_range
        self.active_threshold = active_threshold
        self.steps = steps

    def apply(self, ca: CAState) -> CAState:
        from scipy.ndimage import uniform_filter

        s = (ca.state > self.active_threshold).astype(np.float64)

        for _ in range(self.steps):
            # Sum of 3×3 neighbourhood (includes centre)
            nbr_sum = uniform_filter(s, size=3, mode="wrap") * 9.0
            # Exclude centre: neighbour-only sum
            nbr_sum -= s

            alive = s > 0
            born = (~alive) & (nbr_sum >= self.birth_range[0]) & (nbr_sum <= self.birth_range[1])
            survive = alive & (nbr_sum >= self.survive_range[0]) & (nbr_sum <= self.survive_range[1])
            s = (born | survive).astype(np.float64)

        # Blend with original intensity at active cells
        ca.state = np.where(s > 0, ca.state * s + s * 0.5, 0.0)
        ca.state = np.clip(ca.state / (ca.state.max() + 1e-8), 0.0, 1.0)
        return ca
