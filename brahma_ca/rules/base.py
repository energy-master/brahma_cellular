from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brahma_ca.state import CAState


class CARule(ABC):
    @abstractmethod
    def apply(self, ca: "CAState") -> "CAState":
        ...

    def __or__(self, other: "CARule") -> "RuleChain":
        return RuleChain([self, other])


class RuleChain(CARule):
    def __init__(self, rules: list[CARule]):
        self.rules = rules

    def apply(self, ca: "CAState") -> "CAState":
        for rule in self.rules:
            ca = rule.apply(ca)
        return ca

    def __or__(self, other: "CARule") -> "RuleChain":
        return RuleChain(self.rules + [other])
