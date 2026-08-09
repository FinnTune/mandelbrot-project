import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _resolve_engine(engine):
    if engine == "auto":
        engine = "numba" if _NUMBA_AVAILABLE else "numpy"

    if engine == "numba" and not _NUMBA_AVAILABLE:
        raise ImportError(
            "engine='numba' requires the 'numba' package (pip install numba)"
        )

    if engine not in ("numpy", "numba"):
        raise ValueError(f"Unknown engine {engine!r}; expected 'numpy', 'numba', or 'auto'")

    return engine


# The Mandelbrot and Julia sets share one escape-time recurrence,
# z_{n+1} = z_n**2 + c -- a Mandelbrot render holds z0 fixed at 0 and
# varies c per pixel, a Julia render holds c fixed and varies z0 per
# pixel. Both engines below take a "vary_z0" grid parametrization so the
# iteration logic itself is written once.


def _escape_time_numpy(z0, c, width, height, max_iter):
    z0 = np.broadcast_to(np.asarray(z0, dtype=complex), (width * height,))
    c = np.broadcast_to(np.asarray(c, dtype=complex), (width * height,))
    z = z0.copy()
    divtime = np.full(z.shape, max_iter, dtype=int)

    # Track only still-active (non-escaped) points so later iterations do
    # less work as more of the grid escapes, instead of touching every
    # pixel on every pass.
    active = np.arange(z.size)
    for i in range(max_iter):
        z_active = z[active] ** 2 + c[active]
        z[active] = z_active

        escaped = np.abs(z_active) > 2
        divtime[active[escaped]] = i

        active = active[~escaped]
        if active.size == 0:
            break

    return divtime.reshape(width, height).T


def _escape_time_numpy_smooth(z0, c, width, height, max_iter):
    z0 = np.broadcast_to(np.asarray(z0, dtype=complex), (width * height,))
    c = np.broadcast_to(np.asarray(c, dtype=complex), (width * height,))
    z = z0.copy()
    divtime = np.full(z.shape, float(max_iter), dtype=float)

    active = np.arange(z.size)
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


def _mandelbrot_set_numpy(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c = (x[:, None] + 1j * y[None, :]).ravel()
    return _escape_time_numpy(0j, c, width, height, max_iter)


def _mandelbrot_set_numpy_smooth(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c = (x[:, None] + 1j * y[None, :]).ravel()
    return _escape_time_numpy_smooth(0j, c, width, height, max_iter)


def _julia_set_numpy(c, xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    z0 = (x[:, None] + 1j * y[None, :]).ravel()
    return _escape_time_numpy(z0, c, width, height, max_iter)


def _julia_set_numpy_smooth(c, xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    z0 = (x[:, None] + 1j * y[None, :]).ravel()
    return _escape_time_numpy_smooth(z0, c, width, height, max_iter)


if _NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _escape_time_numba(
        xmin, xmax, ymin, ymax, width, height, max_iter, z0r, z0i, cr, ci, vary_z0
    ):
        divtime = np.full((height, width), max_iter, dtype=np.int64)
        for i in prange(width):
            gx = xmin if width == 1 else xmin + (xmax - xmin) * i / (width - 1)
            for j in range(height):
                gy = ymin if height == 1 else ymin + (ymax - ymin) * j / (height - 1)
                if vary_z0:
                    zr, zi, pr, pi = gx, gy, cr, ci
                else:
                    zr, zi, pr, pi = z0r, z0i, gx, gy
                for k in range(max_iter):
                    zr, zi = zr * zr - zi * zi + pr, 2.0 * zr * zi + pi
                    if zr * zr + zi * zi > 4.0:
                        divtime[j, i] = k
                        break
        return divtime

    @njit(parallel=True, cache=True)
    def _escape_time_numba_smooth(
        xmin, xmax, ymin, ymax, width, height, max_iter, z0r, z0i, cr, ci, vary_z0
    ):
        divtime = np.full((height, width), float(max_iter), dtype=np.float64)
        log2 = np.log(2.0)
        for i in prange(width):
            gx = xmin if width == 1 else xmin + (xmax - xmin) * i / (width - 1)
            for j in range(height):
                gy = ymin if height == 1 else ymin + (ymax - ymin) * j / (height - 1)
                if vary_z0:
                    zr, zi, pr, pi = gx, gy, cr, ci
                else:
                    zr, zi, pr, pi = z0r, z0i, gx, gy
                for k in range(max_iter):
                    zr, zi = zr * zr - zi * zi + pr, 2.0 * zr * zi + pi
                    mag2 = zr * zr + zi * zi
                    if mag2 > 4.0:
                        divtime[j, i] = (k + 1) - np.log(np.log(np.sqrt(mag2))) / log2
                        break
        return divtime

    def _mandelbrot_set_numba(xmin, xmax, ymin, ymax, width, height, max_iter):
        return _escape_time_numba(
            xmin, xmax, ymin, ymax, width, height, max_iter, 0.0, 0.0, 0.0, 0.0, False
        )

    def _mandelbrot_set_numba_smooth(xmin, xmax, ymin, ymax, width, height, max_iter):
        return _escape_time_numba_smooth(
            xmin, xmax, ymin, ymax, width, height, max_iter, 0.0, 0.0, 0.0, 0.0, False
        )

    def _julia_set_numba(c, xmin, xmax, ymin, ymax, width, height, max_iter):
        return _escape_time_numba(
            xmin, xmax, ymin, ymax, width, height, max_iter, 0.0, 0.0, c.real, c.imag, True
        )

    def _julia_set_numba_smooth(c, xmin, xmax, ymin, ymax, width, height, max_iter):
        return _escape_time_numba_smooth(
            xmin, xmax, ymin, ymax, width, height, max_iter, 0.0, 0.0, c.real, c.imag, True
        )


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
    engine = _resolve_engine(engine)
    if engine == "numba":
        fn = _mandelbrot_set_numba_smooth if smooth else _mandelbrot_set_numba
    else:
        fn = _mandelbrot_set_numpy_smooth if smooth else _mandelbrot_set_numpy
    return fn(xmin, xmax, ymin, ymax, width, height, max_iter)


def julia_set(
    c,
    xmin=-1.5,
    xmax=1.5,
    ymin=-1.5,
    ymax=1.5,
    width=800,
    height=600,
    max_iter=100,
    engine="numpy",
    smooth=False,
):
    c = complex(c)
    engine = _resolve_engine(engine)
    if engine == "numba":
        fn = _julia_set_numba_smooth if smooth else _julia_set_numba
    else:
        fn = _julia_set_numpy_smooth if smooth else _julia_set_numpy
    return fn(c, xmin, xmax, ymin, ymax, width, height, max_iter)


def _plot_escape_time(
    data, xmin, xmax, ymin, ymax, max_iter, cmap, smooth, title
):
    plt.figure(figsize=(12, 8))
    plt.imshow(
        data,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        norm=LogNorm(),
    )
    label = "Smooth iteration count (log scale)" if smooth else "Iteration count (log scale)"
    plt.colorbar(label=label)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


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
    _plot_escape_time(
        mandel,
        xmin,
        xmax,
        ymin,
        ymax,
        max_iter,
        cmap,
        smooth,
        title=f"Mandelbrot Set  |  Iterations: {max_iter}",
    )


def plot_julia(
    c=-0.7 + 0.27015j,
    xmin=-1.5,
    xmax=1.5,
    ymin=-1.5,
    ymax=1.5,
    max_iter=100,
    cmap="hot",
    engine="numpy",
    smooth=False,
):
    c = complex(c)
    julia = julia_set(
        c, xmin, xmax, ymin, ymax, max_iter=max_iter, engine=engine, smooth=smooth
    )
    _plot_escape_time(
        julia,
        xmin,
        xmax,
        ymin,
        ymax,
        max_iter,
        cmap,
        smooth,
        title=f"Julia Set  |  c = {c.real:+.5f}{c.imag:+.5f}i  |  Iterations: {max_iter}",
    )
