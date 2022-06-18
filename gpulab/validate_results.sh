#!/bin/bash

#SBATCH -p ffa
#SBATCH -A kdsstudent
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH --mem=8G

set -eu -o pipefail

if [[ $# -lt 2 ]]
then
    echo "Usage: $0 <precision_limit> [group]..." >& 2
    exit 1
fi

REPOSITORY_ROOT_DIR="$(dirname "${PWD}")"

BENCH_DIR="${REPOSITORY_ROOT_DIR}/benchmarking"

LIMIT="$1"
shift

cd "${BENCH_DIR}"

poetry env use python3.9
poetry run python3 -u benchmarking.py validation stats -g "$@" -- "${LIMIT}"
