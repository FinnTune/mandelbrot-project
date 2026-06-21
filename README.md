# Mandelbrot Project

![Test](https://github.com/FinnTune/mandedlbrot-project/actions/workflows/test.yml/badge.svg)

Interactive Mandelbrot set exploration in a Jupyter notebook.

## Description

This project visualizes the Mandelbrot fractal and lets you explore it interactively using sliders and plotting controls in JupyterLab. It combines `numpy` for fast numerical computation, `matplotlib` for rendering, and `ipywidgets` for live parameter tuning so you can zoom into complex regions, adjust iteration depth, and see how small changes reveal intricate self-similar patterns.

## Project files

- `mandelbrot.ipynb`: main notebook with visualization and interactive controls
- `mandelbrot/`: shared Mandelbrot computation and plotting code
- `tests/`: automated tests for core logic, plotting, dependencies, and notebook format
- `requirements.txt`: runtime Python dependencies
- `requirements-dev.txt`: test and CI dependencies
- `Makefile`: shortcuts for environment setup, testing, and notebook launch

## Prerequisites

- Python 3.9+ (or any recent Python 3 version with `venv`)
- `make`

## Quick start

```bash
make install
```

This creates `.venv/`, upgrades `pip`, and installs dependencies.

## Run the project

```bash
make run-notebook
# or
make run-lab
```

This opens JupyterLab. Then open `mandelbrot.ipynb`.

## Run tests

```bash
make test
```

This installs dev dependencies and runs the pytest suite headlessly.

## Continuous integration

GitHub Actions runs on every push and pull request to `master`:

- installs runtime and dev dependencies
- runs `pytest`
- executes `mandelbrot.ipynb` to verify the notebook still runs end-to-end

Dependabot opens weekly pull requests for Python package and GitHub Actions updates.

## Makefile commands

- `make venv` - create local virtual environment and upgrade `pip`
- `make install` - create venv and install `requirements.txt`
- `make install-dev` - install runtime and test dependencies
- `make test` - run the pytest suite
- `make run-notebook` - install dependencies and start JupyterLab
- `make run-lab` - alias for `make run-notebook`
- `make freeze` - write currently installed packages to `requirements.txt`
- `make clean-venv` - remove `.venv/`

## Manual setup (without Makefile)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m jupyter lab
```

## Notes

- The virtual environment directory `.venv/` is already ignored by git.
- If interactive widgets do not render, restart the kernel and rerun all cells.

## License

Copyright (C) 2026 Andre Teetor

This project is licensed under the GNU General Public License v2.0 —
see the [LICENSE](LICENSE) file for details.
