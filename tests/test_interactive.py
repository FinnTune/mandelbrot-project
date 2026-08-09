from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mandelbrot.interactive import ZoomableMandelbrot, zoom_bounds_from_selection


def test_zoom_bounds_from_selection_normalizes_order():
    assert zoom_bounds_from_selection(1.0, -1.0, 2.0, -2.0) == (-1.0, 1.0, -2.0, 2.0)


def test_zoom_bounds_from_selection_rejects_degenerate_rectangle():
    assert zoom_bounds_from_selection(0.5, 0.5, -0.2, 0.3) is None
    assert zoom_bounds_from_selection(0.1, 0.4, 0.2, 0.2) is None


def test_zoomable_mandelbrot_initial_render():
    viewer = ZoomableMandelbrot(max_iter=10)

    assert viewer.bounds == viewer.initial_bounds == (-2.5, 1.0, -1.25, 1.25)
    plt.close(viewer.fig)


def _click_event(x, y, button):
    return SimpleNamespace(xdata=x, ydata=y, button=button)


def test_left_drag_zooms_into_selection():
    viewer = ZoomableMandelbrot(max_iter=10)

    viewer._on_select(_click_event(-1.0, 0.5, 1), _click_event(0.0, -0.5, 1))

    assert viewer.bounds == (-1.0, 0.0, -0.5, 0.5)
    plt.close(viewer.fig)


def test_right_drag_resets_to_initial_view():
    viewer = ZoomableMandelbrot(max_iter=10)
    viewer._on_select(_click_event(-1.0, 0.5, 1), _click_event(0.0, -0.5, 1))
    assert viewer.bounds != viewer.initial_bounds

    viewer._on_select(_click_event(-0.2, 0.1, 3), _click_event(0.2, -0.1, 3))

    assert viewer.bounds == viewer.initial_bounds
    plt.close(viewer.fig)


def test_degenerate_selection_is_ignored():
    viewer = ZoomableMandelbrot(max_iter=10)

    viewer._on_select(_click_event(0.1, 1, 1), _click_event(0.1, 1, 1))

    assert viewer.bounds == viewer.initial_bounds
    plt.close(viewer.fig)
