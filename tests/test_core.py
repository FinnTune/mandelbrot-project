import numpy as np

from mandelbrot.core import mandelbrot_set


def test_output_shape_matches_requested_dimensions():
    result = mandelbrot_set(-2, 1, -1, 1, width=40, height=30, max_iter=20)

    assert result.shape == (30, 40)


def test_origin_remains_in_set():
    result = mandelbrot_set(-0.1, 0.1, -0.1, 0.1, width=11, height=11, max_iter=50)
    center = result[5, 5]

    assert center == 50


def test_point_outside_set_escapes_early():
    result = mandelbrot_set(2.0, 2.0, 0.0, 0.0, width=1, height=1, max_iter=100)

    assert result[0, 0] == 1


def test_iteration_counts_are_within_bounds():
    result = mandelbrot_set(-2.5, 1.0, -1.25, 1.25, width=20, height=20, max_iter=25)

    assert np.all(result >= 0)
    assert np.all(result <= 25)


def test_higher_max_iter_never_reduces_escape_times():
    low_iter = mandelbrot_set(-0.75, -0.74, 0.1, 0.11, width=5, height=5, max_iter=20)
    high_iter = mandelbrot_set(-0.75, -0.74, 0.1, 0.11, width=5, height=5, max_iter=40)

    assert np.all(high_iter >= low_iter)
