
.PHONY: fmt

fmt:
	python -m ruff format
	python -m ruff check --fix --extend-select I


link:
	mkdir -p ${ECS_SCRATCH}/log/bayescl
# 	ln -s ${ECS_SCRATCH}/log/bayescl/ ./log

mypy:
	mkdir -p ${ECS_SCRATCH}/mypy_cache/bayescl
	python -m mypy \
		--sqlite-cache \
		--cache-dir ${ECS_SCRATCH}/mypy_cache/bayescl \
		--ignore-missing-imports \
		--explicit-package-bases .


clean-log:
	rm -rf log/*