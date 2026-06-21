import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mandelbrot.core import plot_mandelbrot


def test_plot_mandelbrot_runs_without_error():
    plot_mandelbrot(max_iter=10)
    plt.close("all")


def test_plot_mandelbrot_accepts_alternate_colormap():
    plot_mandelbrot(max_iter=10, cmap="viridis")
    plt.close("all")
