CONDA_ENV=bayescl

.PHONY: fmt clean-log

fmt:
	python -m ruff format .
	python -m ruff check --fix --extend-select I .

clean-log:
	rm -rv log/*
