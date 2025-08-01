
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


get-data:
	rsync -P lagerfield.ecs.vuw.ac.nz:/local/scratch/antonlee/datasets/core50_128x128.zip /local/scratch/antonlee/datasets

clean-log:
	rm nohup.out || true
	rm -rf log/*

run:
	nohup bash run.sh &
