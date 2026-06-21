import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def mandelbrot_set(xmin, xmax, ymin, ymax, width=800, height=600, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c = x[:, None] + 1j * y[None, :]
    z = np.zeros_like(c)
    divtime = np.full(c.shape, max_iter, dtype=int)

    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]
        divtime[mask & (np.abs(z) > 2)] = i

    return divtime.T


def plot_mandelbrot(
    xmin=-2.5,
    xmax=1.0,
    ymin=-1.25,
    ymax=1.25,
    max_iter=100,
    cmap="hot",
):
    mandel = mandelbrot_set(xmin, xmax, ymin, ymax, max_iter=max_iter)

    plt.figure(figsize=(12, 8))
    plt.imshow(
        mandel,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        norm=LogNorm(),
    )
    plt.colorbar(label="Iteration count (log scale)")
    plt.title(f"Mandelbrot Set  |  Iterations: {max_iter}", fontsize=16, pad=20)
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
