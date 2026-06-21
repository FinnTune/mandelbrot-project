def test_runtime_dependencies_import():
    import ipywidgets
    import matplotlib.pyplot
    import numpy

    assert ipywidgets.__version__
    assert matplotlib.__version__
    assert numpy.__version__


def test_project_package_imports():
    from mandelbrot import mandelbrot_set, plot_mandelbrot

    assert callable(mandelbrot_set)
    assert callable(plot_mandelbrot)
