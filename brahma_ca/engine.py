from __future__ import annotations
import numpy as np
from brahma_ca.state import CAState
from brahma_ca.rules.base import CARule


class CAEngine:
    def __init__(
        self,
        rule: CARule,
        steps: int = 1,
        save_history: bool = False,
        binarize_input: bool = True,
        threshold_pct: float = 80.0,
    ):
        self.rule = rule
        self.steps = steps
        self.save_history = save_history
        self.binarize_input = binarize_input
        self.threshold_pct = threshold_pct

    def evolve(self, ca: CAState) -> CAState:
        if self.binarize_input:
            pct = np.percentile(ca.grid, self.threshold_pct)
            ca.state = (ca.grid > pct).astype(np.float64)

        for _ in range(self.steps):
            if self.save_history:
                ca.history.append(ca.state.copy())
            ca = self.rule.apply(ca)
            ca.step += 1

        return ca

    def reset(self, ca: CAState) -> CAState:
        ca.state = np.zeros_like(ca.grid)
        ca.history = []
        ca.step = 0
        return ca
