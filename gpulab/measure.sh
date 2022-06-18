#!/bin/bash

#SBATCH -p gpu-long
#SBATCH -A kdsstudent
#SBATCH -c 4
#SBATCH --gpus volta:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --requeue

set -eu -o pipefail

if [[ $# -lt 1 ]]
then
    echo "Usage: $0 <benchmark_path> [group]..." >& 2
    exit 1
fi

REPOSITORY_ROOT_DIR="$(dirname "${PWD}")"

BENCH_DIR="${REPOSITORY_ROOT_DIR}/benchmarking"

VALIDATOR_PATH="${REPOSITORY_ROOT_DIR}/existing/python/compute_valid_results.sh"

cd "${BENCH_DIR}"

poetry env use python3.9
poetry run python3 -u benchmarking.py -p "${VALIDATOR_PATH}" benchmark -c "$@"
