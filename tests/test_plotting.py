import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mandelbrot.core import plot_julia, plot_mandelbrot


def test_plot_mandelbrot_runs_without_error():
    plot_mandelbrot(max_iter=10)
    plt.close("all")


def test_plot_mandelbrot_accepts_alternate_colormap():
    plot_mandelbrot(max_iter=10, cmap="viridis")
    plt.close("all")


def test_plot_mandelbrot_accepts_smooth_coloring():
    plot_mandelbrot(max_iter=10, smooth=True)
    plt.close("all")


def test_plot_julia_runs_without_error():
    plot_julia(max_iter=10)
    plt.close("all")


def test_plot_julia_accepts_custom_c_and_smooth_coloring():
    plot_julia(c=0.285 + 0.01j, max_iter=10, smooth=True, cmap="viridis")
    plt.close("all")
