#!/bin/bash

#SBATCH -p gpu-ffa
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --mem=64G

date

REPOSITORY_ROOT_DIR="$(dirname "${PWD}")"

ch-run ~/containers/cuda-12.2.0 -b ${REPOSITORY_ROOT_DIR}:${REPOSITORY_ROOT_DIR} -w -c ${REPOSITORY_ROOT_DIR}/gpulab ./build.sh

date
