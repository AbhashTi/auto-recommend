import math
import pytest
from core.fuzzy_number import (
    TFN,
    weighted_sum,
    fuzzy_positive_ideal,
    fuzzy_negative_ideal,
)


def test_construction_valid():
    t = TFN(0.2, 0.5, 0.8)
    assert t.as_tuple() == (0.2, 0.5, 0.8)


def test_construction_rejects_invalid_order():
    with pytest.raises(ValueError):
        TFN(0.8, 0.5, 0.2)


def test_from_mean_std():
    # mean=0.78, std=0.12 -> (0.66, 0.78, 0.90)
    t = TFN.from_mean_std(0.78, 0.12)
    assert t.l == pytest.approx(0.66)
    assert t.m == pytest.approx(0.78)
    assert t.u == pytest.approx(0.90)


def test_from_mean_std_clipped_respects_bounds():
    # mean=0.95, std=0.12 would push u to 1.07 -- must clip to 1.0
    t = TFN.from_mean_std_clipped(0.95, 0.12, lo=0.0, hi=1.0)
    assert t.u == pytest.approx(1.0)
    assert t.l == pytest.approx(0.83)


def test_addition():
    a = TFN(0.1, 0.2, 0.3)
    b = TFN(0.05, 0.1, 0.15)
    result = a + b
    assert result.as_tuple() == pytest.approx((0.15, 0.3, 0.45))


def test_subtraction_widens_uncertainty():
    a = TFN(0.5, 0.6, 0.7)
    b = TFN(0.1, 0.2, 0.3)
    result = a - b
    # l uses a.l - b.u, u uses a.u - b.l: uncertainty compounds
    assert result.as_tuple() == pytest.approx((0.2, 0.4, 0.6))


def test_scalar_multiplication_positive():
    t = TFN(0.2, 0.4, 0.6)
    result = t * 2
    assert result.as_tuple() == pytest.approx((0.4, 0.8, 1.2))


def test_scalar_multiplication_negative_flips_order():
    t = TFN(0.2, 0.4, 0.6)
    result = t * -1
    assert result.as_tuple() == pytest.approx((-0.6, -0.4, -0.2))


def test_tfn_multiplication_componentwise():
    a = TFN(0.5, 0.6, 0.7)
    b = TFN(0.8, 0.9, 1.0)
    result = a * b
    assert result.as_tuple() == pytest.approx((0.4, 0.54, 0.7))


def test_division_by_zero_raises():
    t = TFN(0.2, 0.4, 0.6)
    with pytest.raises(ZeroDivisionError):
        t / 0


def test_centroid_defuzzification():
    t = TFN(0.6, 0.8, 1.0)
    assert t.centroid() == pytest.approx(0.8)


def test_width_measures_uncertainty():
    narrow = TFN(0.78, 0.80, 0.82)
    wide = TFN(0.5, 0.8, 1.0)
    assert narrow.width() < wide.width()
    assert wide.width() == pytest.approx(0.5)


def test_vertex_distance_zero_for_identical_tfns():
    a = TFN(0.3, 0.5, 0.7)
    assert a.vertex_distance(a) == pytest.approx(0.0)


def test_vertex_distance_symmetric():
    a = TFN(0.3, 0.5, 0.7)
    b = TFN(0.4, 0.6, 0.8)
    assert a.vertex_distance(b) == pytest.approx(b.vertex_distance(a))


def test_weighted_sum():
    tfns = [TFN(0.2, 0.3, 0.4), TFN(0.5, 0.6, 0.7)]
    weights = [0.5, 0.5]
    result = weighted_sum(tfns, weights)
    assert result.as_tuple() == pytest.approx((0.35, 0.45, 0.55))


def test_fuzzy_positive_and_negative_ideal():
    col = [TFN(0.1, 0.2, 0.3), TFN(0.4, 0.5, 0.6), TFN(0.2, 0.3, 0.4)]
    fpis = fuzzy_positive_ideal(col)
    fnis = fuzzy_negative_ideal(col)
    assert fpis.as_tuple() == pytest.approx((0.4, 0.5, 0.6))
    assert fnis.as_tuple() == pytest.approx((0.1, 0.2, 0.3))


def test_worked_example_hyundai_creta_interior_quality():
    """
    Sanity check matching the design doc's worked example style:
    mean sentiment 0.78, std 0.12 across reviews -> TFN(0.66, 0.78, 0.90),
    centroid 0.78, width 0.24 (moderate agreement).
    """
    t = TFN.from_mean_std(mean=0.78, std=0.12)
    assert t.centroid() == pytest.approx(0.78)
    assert t.width() == pytest.approx(0.24)
