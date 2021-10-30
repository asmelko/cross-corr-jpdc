#pragma once

#include "types.cuh"

namespace cross {

template<typename T>
void run_hadamard_original(
    const T* ref,
    T* deformed,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    dsize_t num_threads
);

template<typename T, typename RES>
void run_cross_corr_naive_original(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

template<typename T, typename RES>
void run_ccn_ring_buffer_row(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
);

template<typename T, typename RES>
void run_ccn_def_per_block(
    const T* __restrict__ ref_mat,
    const T* __restrict__ def_mats,
    RES* __restrict__ out_mats,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_def_mats,
    dsize_t total_num_blocks,
    dsize_t threads_per_block
);

}