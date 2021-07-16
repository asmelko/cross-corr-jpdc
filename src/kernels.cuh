#pragma once

#include "types.cuh"

namespace cross {

template<typename T>
void run_hadamard_original(
    T* deformed,
    const T* ref,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    dsize_t num_threads
);

template<typename T, typename RES>
void run_cross_corr_naive_original(
    const T* __restrict__ deformed,
    const T* __restrict__ ref,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

}