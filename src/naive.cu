#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "types.cuh"
#include "helpers.cuh"

namespace cg = cooperative_groups;

namespace cross
{

/**
 * This kernel is a reimplementation of the original naive cross_corr kernel
 * The kernel receives reference subregions, each in row major order all stacked one after another
 * into a single array "ref". "deformed" contains corresponding subregions from "batch_size" of the deformed  pictures
 * which are to be cross-correlated with the reference subregions. All subregions are in row major order, first
 * all subregions of the first deformed image, then all subregions of the second deformed image up to the "batch_size"th
 * deformed image. Number of subregions from the reference and all the deformed images is the same.
 * The input arrays ref and deformed contain only the subregions themselfs, and we must
 * clamp the computation to use only the overlapping parts.
 *
 * For each subregion we search an area of the size "search_size" for cross-correlation maximum.
 * The whole strip of deformed subregions is partitioned into a 16x16 CUDA blocks,
 * where each thread computes one possible shift of the reference image.
 * Output contains an an array of "search_size" results in row major order
 * corresponding to the result of cross correlation for each position in the search area.
 *
 * The memory access patterns are not ideal. Due to the 16x16 size of each block,
 * each half of the warp accesses different row of the "picture", most likely leading to two 128 byte
 * global memory accesses. The implementation also does not use shared memory in any way.
 */
template<typename T, typename RES>
__global__ void cross_corr_naive_original(
    const T* __restrict__ deformed,
    const T* __restrict__ ref,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size

) {
    cg::thread_block ctb = cg::this_thread_block();

    // Coordinates in the whole strip of deformed subregions
    unsigned int def_strip_x = ctb.group_index().x * ctb.group_dim().x + ctb.thread_index().x;
    unsigned int def_strip_y = ctb.group_index().y * ctb.group_dim().y + ctb.thread_index().y;

    unsigned int region_idx = def_strip_x / search_size.x;

    if (region_idx >= subregions_per_pic || def_strip_y >= search_size.y) {
        return;
    }

    // Position of the centre of the subregion
    dsize2_t in_region_pos = { def_strip_x % search_size.x, def_strip_y };
    dsize_t ref_idx = region_idx % subregions_per_pic;
    dsize2_t half_size = (search_size - 1) / 2;

    vec2<int> shift = {(int)in_region_pos.x - (int)half_size.x, (int)in_region_pos.y - (int)half_size.y};

    ref += ref_idx * subregion_size.area();
    deformed += region_idx * subregion_size.area();
    out += region_idx * search_size.area();

    for (dsize_t i = 0; i < batch_size; ++i) {
        dsize_t x_start = max(-shift.x, 0);
        dsize_t x_end = min(subregion_size.x - shift.x, subregion_size.x);
        dsize_t y_start = max(-shift.y, 0);
        dsize_t y_end = min(subregion_size.y - shift.y, subregion_size.y);

        RES sum = 0;
        for (dsize_t y = y_start; y < y_end; ++y) {
            for (dsize_t x = x_start; x < x_end; ++x) {
                int x_shifted = x + shift.x;
                int y_shifted = y + shift.y;

                sum += deformed[y_shifted * subregion_size.x + x_shifted] * ref[y * subregion_size.x + x];
            }
        }

        out[in_region_pos.linear_idx(search_size.x)] = sum;

        deformed += subregions_per_pic * subregion_size.area();
        out += subregions_per_pic * search_size.area();
    }
}

/*
TODO: Try loading the row of reference subregion to shared memory
then do the multiplication of each ref row element with each deformed row element,
either materializing this matrix or computing it during a sum.

As each pair of ref and deformed are multiplied only in a signle shift, it may be better to just
do it during the sum computation.

So with single dimensional blocks, we just load the ref row (or part of it) and
each thread goes linearily through it's deformed row and computes the
multiplication of each element with the corresponding ref element.

Threads iterate this computation over the ref elemens to use shared memory broadcast.
So that each thread in a warp accesses the same ref element and multiplies it with the elements
in its deformed row.

As each thread in a warp will be processing a successive elements of the deformed row,
we could do thread shuffle to reduce the accesses to global memory.




TODO: Try using the whole warp for computation of a single shift
*/

template<typename T, typename RES>
void run_cross_corr_naive_original(
    const T* __restrict__ deformed,
    const T* __restrict__ ref,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
) {
    dim3 num_threads(16, 16);
    dim3 num_blocks(
        div_up(search_size.x * subregions_per_pic, num_threads.x),
        div_up(search_size.y, num_threads.y)
    );

    cross_corr_naive_original<<<num_blocks, num_threads>>>(
        deformed,
        ref,
        out,
        subregion_size,
        search_size,
        subregions_per_pic,
        batch_size
    );
}

template void run_cross_corr_naive_original<int, int>(
    const int* __restrict__ deformed,
    const int* __restrict__ ref,
    int* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

template void run_cross_corr_naive_original<float, float>(
    const float* __restrict__ deformed,
    const float* __restrict__ ref,
    float* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

template void run_cross_corr_naive_original<double, double>(
    const double* __restrict__ deformed,
    const double* __restrict__ ref,
    double* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

}
