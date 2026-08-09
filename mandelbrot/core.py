import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _mandelbrot_set_numpy(xmin, xmax, ymin, ymax, width, height, max_iter):
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


def _mandelbrot_set_numpy_smooth(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c = (x[:, None] + 1j * y[None, :]).ravel()
    z = np.zeros_like(c)
    divtime = np.full(c.shape, float(max_iter), dtype=float)

    active = np.arange(c.size)
    for i in range(max_iter):
        z_active = z[active] ** 2 + c[active]
        z[active] = z_active

        abs_z = np.abs(z_active)
        escaped = abs_z > 2
        if np.any(escaped):
            # Normalized/smooth escape count: interpolates between iteration
            # i and i + 1 using how far past the bailout radius z landed, so
            # neighboring pixels get a continuous gradient instead of a hard
            # iteration-count band.
            divtime[active[escaped]] = (i + 1) - np.log(np.log(abs_z[escaped])) / np.log(2)

        active = active[~escaped]
        if active.size == 0:
            break

    return divtime.reshape(width, height).T


if _NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _mandelbrot_set_numba(xmin, xmax, ymin, ymax, width, height, max_iter):
        divtime = np.full((height, width), max_iter, dtype=np.int64)
        for i in prange(width):
            cr = xmin if width == 1 else xmin + (xmax - xmin) * i / (width - 1)
            for j in range(height):
                ci = ymin if height == 1 else ymin + (ymax - ymin) * j / (height - 1)
                zr = 0.0
                zi = 0.0
                for k in range(max_iter):
                    zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                    if zr * zr + zi * zi > 4.0:
                        divtime[j, i] = k
                        break
        return divtime

    @njit(parallel=True, cache=True)
    def _mandelbrot_set_numba_smooth(xmin, xmax, ymin, ymax, width, height, max_iter):
        divtime = np.full((height, width), float(max_iter), dtype=np.float64)
        log2 = np.log(2.0)
        for i in prange(width):
            cr = xmin if width == 1 else xmin + (xmax - xmin) * i / (width - 1)
            for j in range(height):
                ci = ymin if height == 1 else ymin + (ymax - ymin) * j / (height - 1)
                zr = 0.0
                zi = 0.0
                for k in range(max_iter):
                    zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                    mag2 = zr * zr + zi * zi
                    if mag2 > 4.0:
                        divtime[j, i] = (k + 1) - np.log(np.log(np.sqrt(mag2))) / log2
                        break
        return divtime


def mandelbrot_set(
    xmin,
    xmax,
    ymin,
    ymax,
    width=800,
    height=600,
    max_iter=100,
    engine="numpy",
    smooth=False,
):
    if engine == "auto":
        engine = "numba" if _NUMBA_AVAILABLE else "numpy"

    if engine == "numba":
        if not _NUMBA_AVAILABLE:
            raise ImportError(
                "engine='numba' requires the 'numba' package (pip install numba)"
            )
        fn = _mandelbrot_set_numba_smooth if smooth else _mandelbrot_set_numba
        return fn(xmin, xmax, ymin, ymax, width, height, max_iter)

    if engine != "numpy":
        raise ValueError(f"Unknown engine {engine!r}; expected 'numpy', 'numba', or 'auto'")

    fn = _mandelbrot_set_numpy_smooth if smooth else _mandelbrot_set_numpy
    return fn(xmin, xmax, ymin, ymax, width, height, max_iter)


def plot_mandelbrot(
    xmin=-2.5,
    xmax=1.0,
    ymin=-1.25,
    ymax=1.25,
    max_iter=100,
    cmap="hot",
    engine="numpy",
    smooth=False,
):
    mandel = mandelbrot_set(
        xmin, xmax, ymin, ymax, max_iter=max_iter, engine=engine, smooth=smooth
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(
        mandel,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        norm=LogNorm(),
    )
    label = "Smooth iteration count (log scale)" if smooth else "Iteration count (log scale)"
    plt.colorbar(label=label)
    plt.title(f"Mandelbrot Set  |  Iterations: {max_iter}", fontsize=16, pad=20)
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
