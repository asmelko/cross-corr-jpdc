#!/bin/bash

#SBATCH -p gpu-ffa
#SBATCH --time=12:00:00

date

REPOSITORY_ROOT_DIR="$(dirname "${PWD}")"

ch-run ~/containers/cuda-12.2.0 -b ${REPOSITORY_ROOT_DIR}:${REPOSITORY_ROOT_DIR} -w -c ${REPOSITORY_ROOT_DIR}/visualization -- python3 preprocess.py $1

date
