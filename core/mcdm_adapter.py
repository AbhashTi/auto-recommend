"""
Adapter between our TFN library (core/fuzzy_number.py) and pyfdm's ranking
engines (fTOPSIS, fVIKOR, fAHP, entropy weights).

Why this file exists: pyfdm represents a fuzzy decision matrix as a plain
numpy array of shape (n_alternatives, n_criteria, 3), where the last axis
is (l, m, u). Our own TFN objects are richer (they carry the from_mean_std
construction logic, vertex distance, etc.) but pyfdm just needs the raw
numbers for the ranking step. This module is the one-way bridge:
build your decision matrix with TFN objects (readable, testable, documented),
then convert to a pyfdm array right before ranking.
"""

from __future__ import annotations
import numpy as np
from core.fuzzy_number import TFN


def tfn_matrix_to_pyfdm(matrix: list[list[TFN]]) -> np.ndarray:
    """
    Convert a (n_alternatives x n_criteria) matrix of TFN objects into the
    (n_alternatives, n_criteria, 3) numpy array pyfdm's methods expect.
    """
    return np.array(
        [[tfn.as_tuple() for tfn in row] for row in matrix],
        dtype=float,
    )


def rank_with_topsis(
    matrix: list[list[TFN]],
    weights: list[float],
    types: list[int],
) -> dict:
    """
    Run fuzzy TOPSIS via pyfdm.

    types: 1 for benefit criteria (higher is better, e.g. safety, comfort),
          -1 for cost criteria (lower is better, e.g. price, depreciation).
    weights: crisp weights (from fAHP centroid or entropy weighting),
             summing to 1.
    """
    from pyfdm.methods import fTOPSIS

    pyfdm_matrix = tfn_matrix_to_pyfdm(matrix)
    method = fTOPSIS()
    scores = method(pyfdm_matrix, np.array(weights), np.array(types))
    ranking = np.argsort(-scores)  # descending: highest CC = rank 1
    return {
        "closeness_coefficients": scores.tolist(),
        "ranking_indices": ranking.tolist(),  # index 0 = best car
    }


def rank_with_vikor(
    matrix: list[list[TFN]],
    weights: list[float],
    types: list[int],
    v: float = 0.5,
) -> dict:
    """Run fuzzy VIKOR via pyfdm. Lower Q score = better."""
    from pyfdm.methods import fVIKOR

    pyfdm_matrix = tfn_matrix_to_pyfdm(matrix)
    method = fVIKOR()
    raw = method(pyfdm_matrix, np.array(weights), np.array(types), v=v)
    # fVIKOR returns columns [S, R, Q] per alternative; Q (compromise
    # score, last column) is what ranks alternatives -- lower Q is better.
    q_scores = raw[:, -1]
    ranking = np.argsort(q_scores)  # ascending: lowest Q = rank 1
    return {
        "q_scores": q_scores.tolist(),
        "ranking_indices": ranking.tolist(),
    }


def elicit_weights_fahp(pairwise_tfn_matrix: np.ndarray) -> np.ndarray:
    """
    Run fuzzy AHP on a pairwise TFN comparison matrix (criterion i vs j)
    to get fuzzy weights, then defuzzify (centroid) to crisp weights for
    use in fTOPSIS/fVIKOR.

    pairwise_tfn_matrix shape: (n_criteria, n_criteria, 3) - build this
    from the user's 21-question quiz answers (7 criteria -> C(7,2) = 21
    pairwise comparisons).
    """
    from pyfdm.weights.subjective import fAHP

    ahp = fAHP()
    fuzzy_weights = ahp(pairwise_tfn_matrix)  # shape (n_criteria, 3)
    crisp_weights = fuzzy_weights.mean(axis=-1)  # centroid defuzzification
    return crisp_weights / crisp_weights.sum()  # renormalize to sum to 1


def rank_correlation(ranking_a: list[int], ranking_b: list[int]) -> float:
    """
    Spearman rank correlation between TOPSIS and VIKOR rankings - this is
    the rho >= 0.85 agreement gate from the design doc's evaluation plan.
    """
    from scipy.stats import spearmanr

    corr, _ = spearmanr(ranking_a, ranking_b)
    return float(corr)
