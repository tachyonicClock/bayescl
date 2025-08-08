CONDA_ENV=bayescl

.PHONY: fmt

fmt:
	python -m ruff format
	python -m ruff check --fix --extend-select I


link:
	rm -r log || true
	mkdir -p ${ECS_SCRATCH}/log/bayescl
	ln -s ${ECS_SCRATCH}/log/bayescl/ ./logs

nesi-link:
	rm -r log || true
	mkdir -p ${HOME}/project/log/bayescl
	ln -s ${HOME}/project/log/bayescl ./log

nesi-conda:
	mkdir -p ${HOME}/nobackup/pyvenv
	conda create --prefix ${HOME}/nobackup/pyvenv/bayescl python=3.12
	# Run:
	# 	conda activate ${HOME}/nobackup/pyvenv/${CONDA_ENV}
	#	conda config --set env_prompt ${CONDA_ENV}


pull-nesi:
	rsync -aP nesi:/home/leea6/project/repos/bayescl/log/. ./log

mypy:
	mkdir -p ${ECS_SCRATCH}/mypy_cache/bayescl
	python -m mypy \
		--sqlite-cache \
		--cache-dir ${ECS_SCRATCH}/mypy_cache/bayescl \
		--ignore-missing-imports \
		--explicit-package-bases .


get-data:
	rsync -P lagerfield.ecs.vuw.ac.nz:/local/scratch/antonlee/datasets/core50_128x128.zip /local/scratch/antonlee/datasets
	rsync -P lagerfield.ecs.vuw.ac.nz:${DATASETS}/imagenet-r.tar ${DATASETS}

clean-log:
	rm nohup.out || true
	rm -rf log/*

run:
	nohup bash run.sh &
