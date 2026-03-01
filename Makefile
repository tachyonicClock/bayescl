CONDA_ENV=bayescl

.PHONY: fmt clean-log

fmt:
	python -m ruff format .
	python -m ruff check --fix --extend-select I .

# 	clear notebooks
	find . -type f -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \;

clean-log:
	rm -rv log/*
