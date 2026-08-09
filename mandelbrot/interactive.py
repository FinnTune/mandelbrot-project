import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import RectangleSelector

from mandelbrot.core import mandelbrot_set


def zoom_bounds_from_selection(x0, x1, y0, y1, min_size=1e-12):
    """Normalize a click-drag rectangle (in data coordinates) into
    (xmin, xmax, ymin, ymax). Returns None for degenerate selections
    (e.g. a stray click) so callers can ignore them."""
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))
    if xmax - xmin < min_size or ymax - ymin < min_size:
        return None
    return xmin, xmax, ymin, ymax


class ZoomableMandelbrot:
    """Interactive Mandelbrot viewer: left-drag to zoom into a region,
    right-drag to reset to the initial view."""

    def __init__(
        self,
        xmin=-2.5,
        xmax=1.0,
        ymin=-1.25,
        ymax=1.25,
        max_iter=100,
        cmap="hot",
        engine="numpy",
        smooth=False,
        figsize=(12, 8),
    ):
        self.initial_bounds = (xmin, xmax, ymin, ymax)
        self.bounds = (xmin, xmax, ymin, ymax)
        self.max_iter = max_iter
        self.cmap = cmap
        self.engine = engine
        self.smooth = smooth

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._image = None
        self._draw()

        self.selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1, 3],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )

    def _draw(self):
        xmin, xmax, ymin, ymax = self.bounds
        mandel = mandelbrot_set(
            xmin,
            xmax,
            ymin,
            ymax,
            max_iter=self.max_iter,
            engine=self.engine,
            smooth=self.smooth,
        )

        if self._image is None:
            self._image = self.ax.imshow(
                mandel,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap=self.cmap,
                norm=LogNorm(),
            )
            label = (
                "Smooth iteration count (log scale)"
                if self.smooth
                else "Iteration count (log scale)"
            )
            self.fig.colorbar(self._image, ax=self.ax, label=label)
            self.ax.set_xlabel("Real axis")
            self.ax.set_ylabel("Imaginary axis")
        else:
            self._image.set_data(mandel)
            self._image.set_extent([xmin, xmax, ymin, ymax])
            self._image.autoscale()

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_title(
            f"Mandelbrot Set  |  Iterations: {self.max_iter}  "
            "(drag to zoom, right-drag to reset)",
            fontsize=14,
        )
        self.fig.canvas.draw_idle()

    def _on_select(self, eclick, erelease):
        if eclick.button == 3:
            self.bounds = self.initial_bounds
            self._draw()
            return

        new_bounds = zoom_bounds_from_selection(
            eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata
        )
        if new_bounds is None:
            return
        self.bounds = new_bounds
        self._draw()


def interactive_mandelbrot(**kwargs):
    """Display a live, click-and-drag zoomable Mandelbrot viewer.

    Mirrors plot_mandelbrot's keyword arguments (xmin, xmax, ymin, ymax,
    max_iter, cmap, engine, smooth). Requires an interactive matplotlib
    backend (e.g. `%matplotlib widget` in Jupyter) for the drag gestures
    to register.
    """
    return ZoomableMandelbrot(**kwargs)
