#!/bin/bash

#SBATCH -p gpu-ffa
#SBATCH -c 4
#SBATCH --gpus volta:1
#SBATCH --time=12:00:00

set -eu -o pipefail

REPOSITORY_ROOT_DIR="$(dirname "${PWD}")"

BUILD="${REPOSITORY_ROOT_DIR}/build"
ARGS="${REPOSITORY_ROOT_DIR}/data/args"
DATA="${REPOSITORY_ROOT_DIR}/data/input"

ncu --set full --export "${DIR}/noopt_one_to_one" "${BUILD}/cross" run --args_path "${ARGS}/noopt_one_to_one.json" nai_warp_shuffle_one_to_one "${DATA}/ina_256_256_1_1.csv" "${DATA}/ina_256_256_1_2.csv"
#ncu --set full --export "${DIR}/multirow" "${BUILD}/cross" run --args_path "${ARGS}/multirow.json" nai_multirow_shuffle_one_to_one "${DATA}/ina_256_256_1_1.csv" "${DATA}/ina_256_256_1_2.csv"
#ncu --set full --export "${DIR}/noopt_one_to_many" "${BUILD}/cross" run --args_path "${ARGS}/noopt_one_to_many.json" nai_warp_shuffle_one_to_many "${DATA}/ina_256_256_1_1.csv" "${DATA}/ina_256_256_16_1.csv"
#ncu --set full --export "${DIR}/multiright" "${BUILD}/cross" run --args_path "${ARGS}/multiright.json" nai_warp_shuffle_one_to_many "${DATA}/ina_256_256_1_1.csv" "${DATA}/ina_256_256_16_1.csv"

