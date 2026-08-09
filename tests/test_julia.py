import numpy as np
import pytest

from mandelbrot.core import julia_set


def test_output_shape_matches_requested_dimensions():
    result = julia_set(
        -0.7 + 0.27015j, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, width=40, height=30, max_iter=20
    )

    assert result.shape == (30, 40)


def test_bounded_orbit_never_escapes():
    # c=0 makes the recurrence trivial (z_{n+1} = z_n**2), which escapes
    # iff |z0| > 1. z0=0.5 stays inside the unit disk forever.
    result = julia_set(0, xmin=0.5, xmax=0.5, ymin=0, ymax=0, width=1, height=1, max_iter=50)

    assert result[0, 0] == 50


def test_point_outside_set_escapes_immediately():
    # Same trivial c=0 recurrence: |z0|=2 > 1, so z1 = 4 already exceeds
    # the bailout radius.
    result = julia_set(0, xmin=2, xmax=2, ymin=0, ymax=0, width=1, height=1, max_iter=50)

    assert result[0, 0] == 0


def test_iteration_counts_are_within_bounds():
    result = julia_set(
        -0.7 + 0.27015j, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, width=20, height=20, max_iter=25
    )

    assert np.all(result >= 0)
    assert np.all(result <= 25)


def test_invalid_engine_raises():
    with pytest.raises(ValueError):
        julia_set(0, xmin=-1, xmax=1, ymin=-1, ymax=1, width=5, height=5, max_iter=10, engine="bogus")


def test_numba_engine_missing_raises_import_error(monkeypatch):
    import mandelbrot.core as core

    monkeypatch.setattr(core, "_NUMBA_AVAILABLE", False)

    with pytest.raises(ImportError):
        julia_set(0, xmin=-1, xmax=1, ymin=-1, ymax=1, width=5, height=5, max_iter=10, engine="numba")


def test_numba_engine_matches_numpy_engine():
    numba = pytest.importorskip("numba")
    del numba

    numpy_result = julia_set(
        -0.7 + 0.27015j, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, width=60, height=45, max_iter=80,
        engine="numpy",
    )
    numba_result = julia_set(
        -0.7 + 0.27015j, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, width=60, height=45, max_iter=80,
        engine="numba",
    )

    assert np.array_equal(numpy_result, numba_result)


def test_smooth_bounded_orbit_stays_at_max_iter():
    result = julia_set(
        0, xmin=0.5, xmax=0.5, ymin=0, ymax=0, width=1, height=1, max_iter=50, smooth=True
    )

    assert result[0, 0] == 50.0


def test_smooth_escaping_point_is_close_to_discrete_value():
    discrete = julia_set(0, xmin=2, xmax=2, ymin=0, ymax=0, width=1, height=1, max_iter=50)
    smooth = julia_set(0, xmin=2, xmax=2, ymin=0, ymax=0, width=1, height=1, max_iter=50, smooth=True)

    assert abs(float(smooth[0, 0]) - float(discrete[0, 0])) <= 2


def test_smooth_numba_engine_matches_numpy_engine():
    numba = pytest.importorskip("numba")
    del numba

    numpy_result = julia_set(
        -0.7 + 0.27015j, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, width=60, height=45, max_iter=80,
        engine="numpy", smooth=True,
    )
    numba_result = julia_set(
        -0.7 + 0.27015j, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, width=60, height=45, max_iter=80,
        engine="numba", smooth=True,
    )

    assert np.allclose(numpy_result, numba_result)


def test_different_c_values_produce_different_sets():
    a = julia_set(-0.7 + 0.27015j, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, width=30, height=30, max_iter=30)
    b = julia_set(0.285 + 0.01j, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, width=30, height=30, max_iter=30)

    assert not np.array_equal(a, b)
