#!/bin/bash

sbatch -w volta04 benchmark.sh ../build/cross volta one_to_one
sbatch -w volta04 benchmark.sh ../build_satur/cross volta one_to_one_saturated
sbatch -w volta04 benchmark.sh ../build/cross volta one_to_many
sbatch -w volta04 benchmark.sh ../build/cross volta one_to_many_saturated
sbatch -w volta04 benchmark.sh ../build/cross volta n_to_mn
sbatch -w volta04 benchmark.sh ../build/cross volta n_to_mn_saturated
sbatch -w volta04 benchmark.sh ../build/cross volta n_to_m
sbatch -w volta04 benchmark.sh ../build/cross volta n_to_m_saturated

sbatch -w ampere01 benchmark.sh ../build/cross ampere one_to_one
sbatch -w ampere01 benchmark.sh ../build_satur/cross ampere one_to_one_saturated
sbatch -w ampere01 benchmark.sh ../build/cross ampere one_to_many
sbatch -w ampere01 benchmark.sh ../build/cross ampere one_to_many_saturated
sbatch -w ampere01 benchmark.sh ../build/cross ampere n_to_mn
sbatch -w ampere01 benchmark.sh ../build/cross ampere n_to_mn_saturated
sbatch -w ampere01 benchmark.sh ../build/cross ampere n_to_m
sbatch -w ampere01 benchmark.sh ../build/cross ampere n_to_m_saturated

sbatch -w ampere02 benchmark.sh ../build/cross ada one_to_one
sbatch -w ampere02 benchmark.sh ../build_satur/cross ada one_to_one_saturated
sbatch -w ampere02 benchmark.sh ../build/cross ada one_to_many
sbatch -w ampere02 benchmark.sh ../build/cross ada one_to_many_saturated
sbatch -w ampere02 benchmark.sh ../build/cross ada n_to_mn
sbatch -w ampere02 benchmark.sh ../build/cross ada n_to_mn_saturated
sbatch -w ampere02 benchmark.sh ../build/cross ada n_to_m
sbatch -w ampere02 benchmark.sh ../build/cross ada n_to_m_saturated
