CONDA_ENV=bayescl

.PHONY: fmt

fmt:
	python -m ruff format
	python -m ruff check --fix --extend-select I


link:
	rm -r logs
	ln -s ${ECS_SCRATCH}/log/bayescl/ ./logs
	mkdir -p ${ECS_SCRATCH}/logs/bayescl

nesi-link:
	rm -r logs 
	mkdir -p ${HOME}/nobackup/logs/bayescl
	ln -s ${HOME}/nobackup/logs/bayescl ./logs

nesi-conda:
	mkdir -p ${HOME}/nobackup/pyvenv
	conda create --prefix ${HOME}/nobackup/pyvenv/bayescl python=3.12
	# Run:
	# 	conda activate ${HOME}/nobackup/pyvenv/${CONDA_ENV}
	#	conda config --set env_prompt ${CONDA_ENV}

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
