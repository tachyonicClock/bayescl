CONDA_ENV=bayescl

.PHONY: fmt

fmt:
	python -m ruff format .
	python -m ruff check --fix --extend-select I .
