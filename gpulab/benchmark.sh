#!/bin/bash

#SBATCH -p gpu-long
#SBATCH -A kdss
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclusive

set -eu -o pipefail

REPOSITORY_ROOT_DIR="$(dirname "${PWD}")"

ch-run ~/containers/cuda-12.2.0 -b ${REPOSITORY_ROOT_DIR}:${REPOSITORY_ROOT_DIR} -w -c ${REPOSITORY_ROOT_DIR}/benchmarking -- ./benchmarking.py -e "$1" benchmark -c -o $2 ./jpdc/benchmark.yml $3
