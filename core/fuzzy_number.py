"""
Triangular Fuzzy Number (TFN) library.

A TFN is represented as (l, m, u): lowest plausible, most likely, highest
plausible. This is the single representation every downstream module
(fuzzy AHP, fuzzy TOPSIS, fuzzy VIKOR) consumes, regardless of whether the
value came from a spec sheet, a crash-test rating, or review sentiment.

Design choices (documented so you can defend them in the viva):
- Multiplication/division are componentwise approximations, not exact TFN
  arithmetic (the true product of two TFNs is a curve, not a triangle).
  This is standard practice in fuzzy MCDM literature and is accurate for
  moderate positive ranges. Keep multiplication chains short (2 deep max)
  to control error accumulation.
- Defuzzification uses the centroid method (mean of l, m, u), which is the
  most common choice in fuzzy TOPSIS/VIKOR implementations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math


@dataclass(frozen=True)
class TFN:
    l: float  # lowest plausible value
    m: float  # most likely value
    u: float  # highest plausible value

    def __post_init__(self):
        if not (self.l <= self.m <= self.u):
            raise ValueError(
                f"TFN must satisfy l <= m <= u, got ({self.l}, {self.m}, {self.u})"
            )

    # ---------- construction helpers ----------

    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> "TFN":
        """
        Build a TFN from an aggregated (mean, std) pair — this is the core
        move for turning NLP sentiment scores into fuzzy numbers.
        TFN = (mean - std, mean, mean + std), clipped to a sane range if
        the caller expects bounded scores (e.g. sentiment in [0, 1]).
        """
        return cls(mean - std, mean, mean + std)

    @classmethod
    def from_mean_std_clipped(
        cls, mean: float, std: float, lo: float = 0.0, hi: float = 1.0
    ) -> "TFN":
        """Same as from_mean_std, but clips l and u into [lo, hi]."""
        l = max(lo, mean - std)
        u = min(hi, mean + std)
        m = min(max(mean, l), u)
        return cls(l, m, u)

    @classmethod
    def crisp(cls, value: float) -> "TFN":
        """A crisp (non-fuzzy) number represented as a degenerate TFN."""
        return cls(value, value, value)

    # ---------- arithmetic ----------

    def __add__(self, other: "TFN") -> "TFN":
        return TFN(self.l + other.l, self.m + other.m, self.u + other.u)

    def __sub__(self, other: "TFN") -> "TFN":
        # Note: subtraction of TFNs widens the result (uncertainty compounds),
        # unlike scalar subtraction.
        return TFN(self.l - other.u, self.m - other.m, self.u - other.l)

    def __mul__(self, other) -> "TFN":
        if isinstance(other, (int, float)):
            if other >= 0:
                return TFN(self.l * other, self.m * other, self.u * other)
            return TFN(self.u * other, self.m * other, self.l * other)
        # componentwise approximation (standard in fuzzy MCDM practice)
        return TFN(self.l * other.l, self.m * other.m, self.u * other.u)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "TFN":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide TFN by zero")
        if scalar > 0:
            return TFN(self.l / scalar, self.m / scalar, self.u / scalar)
        return TFN(self.u / scalar, self.m / scalar, self.l / scalar)

    # ---------- distance / comparison ----------

    def vertex_distance(self, other: "TFN") -> float:
        """
        Vertex method distance between two TFNs — the standard distance
        measure used inside fuzzy TOPSIS to compute closeness to the
        fuzzy positive/negative ideal solutions.
        """
        return math.sqrt(
            ((self.l - other.l) ** 2 + (self.m - other.m) ** 2 + (self.u - other.u) ** 2)
            / 3
        )

    # ---------- defuzzification ----------

    def centroid(self) -> float:
        """Defuzzify to a single crisp value via the centroid (mean) method."""
        return (self.l + self.m + self.u) / 3

    def width(self) -> float:
        """
        The 'uncertainty' of this number. A wide TFN means reviewers
        disagreed (or the spec value has genuine spread); a narrow one
        means consensus. Report this alongside any rank as a confidence
        indicator.
        """
        return self.u - self.l

    # ---------- misc ----------

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.l, self.m, self.u)

    def __repr__(self) -> str:
        return f"TFN({self.l:.4f}, {self.m:.4f}, {self.u:.4f})"


def weighted_sum(tfns: Iterable[TFN], weights: Iterable[float]) -> TFN:
    """Sum of weight_i * tfn_i — used when aggregating criterion scores."""
    total = None
    for tfn, w in zip(tfns, weights):
        term = tfn * w
        total = term if total is None else total + term
    if total is None:
        raise ValueError("weighted_sum requires at least one (tfn, weight) pair")
    return total


def fuzzy_positive_ideal(matrix_column: Iterable[TFN]) -> TFN:
    """
    Fuzzy positive ideal solution (FPIS) for a benefit criterion column:
    the componentwise max across l, m, u seen in that column.
    For cost criteria, use fuzzy_negative_ideal on the same column instead.
    """
    col = list(matrix_column)
    return TFN(
        max(t.l for t in col), max(t.m for t in col), max(t.u for t in col)
    )


def fuzzy_negative_ideal(matrix_column: Iterable[TFN]) -> TFN:
    """Fuzzy negative ideal solution (FNIS): componentwise min."""
    col = list(matrix_column)
    return TFN(
        min(t.l for t in col), min(t.m for t in col), min(t.u for t in col)
    )
