import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def mandelbrot_set(xmin, xmax, ymin, ymax, width=800, height=600, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c = (x[:, None] + 1j * y[None, :]).ravel()
    z = np.zeros_like(c)
    divtime = np.full(c.shape, max_iter, dtype=int)

    # Track only still-active (non-escaped) points so later iterations do
    # less work as more of the grid escapes, instead of touching every
    # pixel on every pass.
    active = np.arange(c.size)
    for i in range(max_iter):
        z_active = z[active] ** 2 + c[active]
        z[active] = z_active

        escaped = np.abs(z_active) > 2
        divtime[active[escaped]] = i

        active = active[~escaped]
        if active.size == 0:
            break

    return divtime.reshape(width, height).T


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
