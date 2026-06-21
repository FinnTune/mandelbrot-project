PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install install-dev freeze clean-venv run-notebook run-lab test

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

test: install-dev
	MPLBACKEND=Agg $(PY) -m pytest

freeze:
	$(PIP) freeze > requirements.txt

run-notebook: install
	$(PY) -m jupyter lab

run-lab: run-notebook

clean-venv:
	rm -rf $(VENV)
