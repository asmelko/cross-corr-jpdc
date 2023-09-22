#!/bin/bash

#SBATCH -p gpu-ffa
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --mem=64G

set -eu -o pipefail

REPOSITORY_ROOT_DIR="$(dirname "${PWD}")"

BUILD_DIR="${REPOSITORY_ROOT_DIR}/build_satur"

mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"

cmake -D USE_SUPERBUILD=OFF -D CMAKE_BUILD_TYPE:STRING=Release -D CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -S "${REPOSITORY_ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --config Release --parallel 8
