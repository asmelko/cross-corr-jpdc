#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "types.cuh"
#include "helpers.cuh"

namespace cg = cooperative_groups;

namespace cross
{

template<typename T>
__global__ void hadamard_original(
    T* __restrict__ deformed,
    const T* __restrict__ ref,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
) {
    cg::thread_block ctb = cg::this_thread_block();

    // Coordinates in the whole strip of deformed subregions
    unsigned int ref_idx = ctb.group_index().x * ctb.group_dim().x + ctb.thread_index().x;

    if (ref_idx >= subregions_per_pic * subregion_size.area()) {
        return;
    }

    for (dsize_t i = ref_idx; i < subregion_size.area() * subregions_per_pic * batch_size; i += subregion_size.area() * subregions_per_pic) {
        // Complex multiplication of complex conjugate of "ref" with "deformed"
        deformed[i] = {
            ref[ref_idx].x * deformed[i].x + ref[ref_idx].y * deformed[i].y,
            ref[ref_idx].x * deformed[i].y - ref[ref_idx].y * deformed[i].x
        };
    }
}

template<typename T>
void run_hadamard_original(
    T* deformed,
    const T* ref,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    dsize_t num_threads
) {
    dsize_t num_blocks = div_up(subregion_size.area() * subregions_per_pic, num_threads);
    hadamard_original<<<num_blocks, num_threads>>>(deformed, ref, subregion_size, subregions_per_pic, batch_size);
}

}