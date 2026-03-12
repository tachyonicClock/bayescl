CONDA_ENV=bayescl

.PHONY: fmt clean-log jsonnet

fmt:
	python -m ruff format .
	python -m ruff check --fix --extend-select I .

# 	run jsonnetfmt on all .jsonnet files
	find . -type f -name "*.jsonnet" -exec jsonnetfmt -i {} \;

# 	clear notebooks
	find . -type f -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \;

clean-log:
	rm -rv log/*


jsonnet:
# Verify that all .jsonnet files can be compiled to JSON
	find . -type f -name "*.jsonnet" -exec jsonnet {} > /dev/null \;
