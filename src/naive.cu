#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "types.cuh"
#include "helpers.cuh"
#include "shared_mem.cuh"

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
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
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
        // The code is different from the original as here we are sliding the
        // deformed region over the reference region, whereas the original
        // did it the other way, which is incorrect in my opinion
        // or at least inconsistent with the text of the thesis
        // where it is defined as reference * deformed
        // and the algorithm clearly states that this means sliding the deformed
        //
        // The results also now match the results of matlab xcorr2
        dsize_t x_ref_start = max(shift.x, 0);
        dsize_t x_ref_end = min(subregion_size.x + shift.x, subregion_size.x);
        dsize_t y_ref_start = max(shift.y, 0);
        dsize_t y_ref_end = min(subregion_size.y + shift.y, subregion_size.y);

        RES sum = 0;
        for (dsize_t y_ref = y_ref_start; y_ref < y_ref_end; ++y_ref) {
            for (dsize_t x_ref = x_ref_start; x_ref < x_ref_end; ++x_ref) {
                // If deformed is shifted by -10, the we are starting from [0,0] in ref
                // and need to start from [10,10] in deformed, as there are 10
                // values to the left and on top outside the reference matrix
                int x_shifted = x_ref - shift.x;
                int y_shifted = y_ref - shift.y;

                sum += deformed[y_shifted * subregion_size.x + x_shifted] * ref[y_ref * subregion_size.x + x_ref];
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

template<typename T>
__device__ dsize_t load_row_chunk(cg::thread_block ctb, T* dst, const T* src, dsize_t row_start, dsize_t row_size, dsize_t chunk_start, dsize_t chunk_size) {
    dsize_t copy_size = min(row_size - chunk_start, chunk_size);
    for (; chunk_start + ctb.thread_index().x < copy_size; chunk_start += ctb.size()) {
        dst[ctb.thread_index().x] = src[row_start + chunk_start + ctb.thread_index().x];
    }
    return copy_size;
}

/** Limited to one reference subregion and one deformed subregion
 *
 * This is just a first simple implementation, future implementations will build on top
 * of this and optimize it
 * Each block processes one row of the ref matrix
 *
 * Each block loads a <block_size> number of elements from a row of the reference subregion and
 * then goes through the deformed subregion, loading it in <block_size> chunks into shared memory.
 *
 * Each
 */
// template<typename T, typename RES>
// __global__ void ccn_shared_mem_rows(
//     const T* __restrict__ ref,
//     const T* __restrict__ deformed,
//     RES* __restrict__ out,
//     dsize2_t subregion_size,
//     dsize2_t search_size
// ) {
//     cg::thread_block ctb = cg::this_thread_block();

//     const dsize_t ref_chunk_size = ctb.size();
//     const dsize_t def_chunk_size = ctb.size();

//     // Split shared memory
//     extern __shared__ T shared[];
//     T* ref_s = shared;
//     T* def_s = ref_s + ref_chunk_size;
//     RES* shift_sums = def_s + 2*def_chunk_size + ref_chunk_size;

//     dsize2_t half_search_size = (search_size - 1) / 2;

//     const dsize_t ref_row = ctb.group_index();
//     const dsize_t ref_row_start = ref_row * subregion_size.x;

//     for (dsize_t ref_chunk_start = 0; ref_chunk_start < subregion_size.x; ref_chunk_start += ref_chunk_size) {
//         // Load part of the ref row into shared memory
//         load_row_chunk(ctb, ref_s, ref, ref_row_start, subregion_size.x, ref_chunk_start, ref_chunk_size);
//         ctb.sync();

//         // Load ref chunk into thread registers
//         // T ref_val = ref_chunk_start + ctb.thread_index().x < subregion_size.x ?
//         //     ref[ref_row_start + ref_chunk_start + ctb.thread_index().x] :
//         //     0;

//         // First deformed row to compute cc with
//         const dsize_t def_begin_row = max((int)ref_row - (int)half_search_size.y, 0);
//         const dsize_t def_end_row = min(ref_row + half_search_size.y, subregion_size.y);
//         for (dsize_t def_row = def_begin_row; def_row < def_end_row; ++def_row) {

//             const dsize_t def_row_start = def_row * subregion_size.x;
//             // For given ref row chunk, we only need to load parts of the def row
//             // that are at most search_size.x before or after
//             const dsize_t def_row_part_start = max(def_row_start, (int)def_row_start + ref_chunk_start - half_search_size.x);
//             const dsize_t def_row_part_end = min(def_row_start + subregion_size.x, def_row_part_start + ref_chunk_size + search_size.x);
//             const dsize_t def_row_part_size = def_row_part_end - def_row_part_start;

//             for (dsize_t def_chunk_start = 0; def_chunk_start < def_row_part_size; def_chunk_start += def_chunk_size) {
//                 load_row_chunk(ctb, def_s, deformed, def_row_part_start, def_row_part_size, def_chunk_start, def_chunk_size);
//                 ctb.sync();



//                 auto ctw = cg::tiled_partition<32>(ctb);
//                 auto num_warps = ctw.meta_group_size();
//                 RES shift_sum = 0;
//                 for (dsize_t shift = 0; shift < 32; ++shift) {

//                 }

//                 // TODO: Add the number of threads as param


//             }
//         }
//     }
// }

/** Two part ring buffer
 *
 * This buffer holds two submatricies, which should be neighbours in the source matrix.
 * When loading, the older submatrix is overwritten
 *
 * Matricies are stored in row major order and should be continuous in the x axis
 *
 * Currently this datastructure is designed to work with blocks with 32x32 threads.
 */
// template<typename T>
// class ring_buffer {
// public:
//     __host__ __device__ ring_buffer(T* data, dsize2_t size)
//         :data_(data), size_(size)
//     {

//     }

//     __device__ void load(cg::thread_block ctb, const T* src, dsize2_t pos) {

//     }

//     __host__ __device__ T operator[](dsize2_t pos) {

//     }
// private:
//     T* data_;
//     dsize2_t pos_;
//     dsize2_t size_;
// };

dsize_t __device__ get_x_def_block_start(dsize_t block_start_x, int max_block_shift_x) {
    return max((int)block_start_x - max_block_shift_x, 0);
}


/**
 * This kernel takes advantage of the access pattern in the original implementation
 * where threads in the same thread block read 15/16 of the same values as different
 * threads from the same thread block in previous step. In the original implementation
 * it is done in the form of 16x16 matrix, which shares 15 columns with the previous matrix.
 *
 * Here we use 1D thread block, so each step would share all but one value from ref and def
 * matricies from the previous step. We use this to preload the next thread block worth of
 * data in one step to shared memory and use this to prevent repeated accesses to global memory,
 * even if those might have been cached.
 *
 * Whereas all threads in a single block access the same <thread_block_size> values from ref,
 * they start at <thread_block_size> different values in the corresponding def row and read
 * another <thread_block_size> values from their initial value. This means that each loop,
 * the whole block accesses <thread_block_size> values from ref but 2x<thread_block_size> values
 * from def. Important note here is that half of the def values is accessed in the next loop too,
 * so we need something like a ring buffer where each loop we load just half of the 2x<thread_block_size>
 * values, allowing us to do this load in one step.
 *
 */
template<typename T, typename RES>
__global__ void ccn_ring_buffer_row(
    const T* __restrict__ ref,
    const T* __restrict__ def,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size
) {
    // TODO: Check block-wide shuffle down from cooperative groups


    cg::thread_block ctb = cg::this_thread_block();

    // Each offset between ref and def is computed by a single thread
    // so each thread corresponds to a single value in the resulting matrix
    // same as the original
    unsigned int res_x = ctb.group_index().x * ctb.group_dim().x + ctb.thread_index().x;

    // As this kernel is simplified and only has 1D thread blocks, each thread block
    // corresponds to a single row
    unsigned int res_y = ctb.group_index().y * ctb.group_dim().y;

    // Index of the leftmost thread in the block
    unsigned int block_res_x = ctb.group_index().x * ctb.group_dim().x;

    T* shared = shared_memory_proxy<T>();
    // All threads read from the same <thread_block_size> values of ref
    // mostly using broadcast, but multiplying them with different but continuous
    // values from def matrix
    // If perfectly aligned, they would all do broadcast from ref and then each read
    // different value from def, where consecutive threads would read consecutive values
    T* ref_s = shared;
    T* def_s = ref_s + ctb.size();

    bool load_bottom = false;

    dsize2_t half_size = (search_size - 1) / 2;
    // As thread indicies are only positive, and half of the resulting matrix
    // represents negative shifts, we need to shift by half the search size to left
    vec2<int> shift{(int)res_x - (int)half_size.x, (int)res_y - (int)half_size.y};
    vec2<int> min_block_shift{(int)block_res_x - (int)half_size.x, (int)res_y - (int)half_size.y};
    vec2<int> max_block_shift{min_block_shift.x + ctb.size(), min_block_shift.y};

    dsize_t x_ref_start = max(shift.x, 0);
    dsize_t x_ref_end = min(subregion_size.x + shift.x, subregion_size.x);
    dsize_t y_ref_start = max(shift.y, 0);
    dsize_t y_ref_end = min(subregion_size.y + shift.y, subregion_size.y);

    dsize_t x_ref_block_start = max(min_block_shift.x, 0);
    dsize_t x_ref_block_end = min(subregion_size.x + max_block_shift.x, subregion_size.x);

    // // Debug
    // if (res_x < search_size.x) {
    //     out[res_y * search_size.x + res_x] = 0;
    // }

    RES sum = 0;
    for (dsize_t y_ref = y_ref_start; y_ref < y_ref_end; ++y_ref) {
        int y_shifted = y_ref - shift.y;

        // Load the initial half of the def buffer
        dsize_t def_top_block_start = max((int)x_ref_block_start - max_block_shift.x, 0);
        dsize_t def_row_idx = def_top_block_start + ctb.thread_index().x;
        // Range check for the load, rest of the code should naturally access only the loaded part
        if (def_row_idx < subregion_size.x) {
            def_s[ctb.thread_index().x] = def[y_shifted * subregion_size.x + def_row_idx];
        }

        load_bottom = false;

        for (dsize_t x_ref_block = x_ref_block_start; x_ref_block < x_ref_block_end; x_ref_block += ctb.size()) {
            dsize_t ref_row_idx = x_ref_block + ctb.thread_index().x;
            if (ref_row_idx < subregion_size.x) {
                ref_s[ctb.thread_index().x] = ref[y_ref * subregion_size.x + ref_row_idx];
            }

            // + ctb.size() as we are prereading the next block of the def matrix after
            // the one of ctb.size() read initialy
            dsize_t def_base_block_start = def_top_block_start;
            def_top_block_start = max((int)x_ref_block - max_block_shift.x, 0) + ctb.size();
            dsize_t def_row_idx = def_top_block_start + ctb.thread_index().x;
            if (def_row_idx < subregion_size.x) {
                T* def_tgt = load_bottom ? def_s : def_s + ctb.size();
                def_tgt[ctb.thread_index().x] = def[y_shifted * subregion_size.x + def_row_idx];
            }
            // If we loaded bottom part, then the older part is in the top part of the buffer
            // so the base needs to be shifted ctb.size() to the left, basically to the middle
            // of the buffer
            int def_base_offset = def_base_block_start - x_ref_block +
                (load_bottom ? ctb.size() : 0);

            load_bottom = !load_bottom;

            ctb.sync();

            // out[res_y * search_size.x + res_x] = x_ref_block;
            // return;

            // if (ctb.thread_index().x == 0) {
            //     for (dsize_t i = 0; i < 10; ++i) {
            //         out[res_y * search_size.x + i] = def_s[i];
            //     }
            // }
            // return ;

            dsize_t start = max((int)x_ref_start - (int)x_ref_block, 0);
            dsize_t end = min((int)x_ref_end - (int)x_ref_block, ctb.size());

            // if (res_x < search_size.x) {
            //     out[res_y * search_size.x + res_x] = end;
            //     return;
            // }
            for (dsize_t ref_s_index = start; ref_s_index < end; ++ref_s_index) {
                // if (res_x < search_size.x) {
                //     out[res_y * search_size.x + res_x] += 1;
                // }
                // if (res_x < search_size.x) {
                    // out[res_y * search_size.x + res_x] = (ref_s_index - (shift.x - def_base_offset)) % 2*ctb.size();
                    // out[res_y * search_size.x + res_x] = (int)ref_s_index - (shift.x - def_base_offset) % (int)(2*ctb.size());
                    // out[res_y * search_size.x + res_x] = (int)ref_s_index - (shift.x - def_base_offset);
                    // out[res_y * search_size.x + res_x] = shift.x;
                    // out[res_y * search_size.x + res_x] = def_base_offset;
                    //return;
                // }

                sum += ref_s[ref_s_index] * def_s[(int)ref_s_index - (shift.x - def_base_offset) % (int)(2*ctb.size())];

            }

            ctb.sync();
        }
    }

    if (res_x < search_size.x) {
        out[res_y * search_size.x + res_x] = sum;
    }

}

/** Ring buffers
 *
 * The original implementation does pretty chaotic memory accesses and code divergence due to the x_ref and y_ref clamping
 *
 * We would like to manipulate the original implementation so that the whole 16x16 threadblock accesses two 16x16
 * submatrices each cycle and does elementwise multiplication, where each thread gets
 * one result element and adds it to its private sum
 *
 * The following cycle the block accesses two 16x16 matrices, which each share
 * 16x15 submatrix with the previous matrix. We could basically just do a ring buffer,
 * where the first warp adds to the ringbuffer each cycle, overwriting the oldest data
 *

 */
// template<typename T, typename RES>
// __global__ void ccn_ring_buffer_matrix(
//     const T* __restrict__ ref,
//     const T* __restrict__ deformed,
//     RES* __restrict__ out,
//     dsize2_t subregion_size,
//     dsize2_t search_size
// ) {
//     // TODO: Check block-wide shuffle down from cooperative groups


//     cg::thread_block ctb = cg::this_thread_block();

//     const dsize_t buffer_size = ctb.size() * 2;

//     extern __shared__ T shared[];

//     //
//     for (dsize_t )
// }


template<typename T, typename RES>
void run_cross_corr_naive_original(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
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
        ref,
        deformed,
        out,
        subregion_size,
        search_size,
        subregions_per_pic,
        batch_size
    );
}

template<typename T, typename RES>
void run_ccn_ring_buffer_row(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
) {
    // TODO: Launch multiple kernels
    dim3 num_blocks(
        div_up(search_size.x, threads_per_block),
        search_size.y
    );

    std::cout << "[" << num_blocks.x << ", " << num_blocks.y << "]\n";

    dsize_t shared_mem_size = 3 * threads_per_block * sizeof(T);

    ccn_ring_buffer_row<<<num_blocks, threads_per_block, shared_mem_size>>>(
        ref,
        deformed,
        out,
        subregion_size,
        search_size
    );
}

template void run_cross_corr_naive_original<int, int>(
    const int* __restrict__ ref,
    const int* __restrict__ deformed,
    int* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

template void run_cross_corr_naive_original<float, float>(
    const float* __restrict__ ref,
    const float* __restrict__ deformed,
    float* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

template void run_cross_corr_naive_original<double, double>(
    const double* __restrict__ ref,
    const double* __restrict__ deformed,
    double* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

template void run_ccn_ring_buffer_row<int, int>(
    const int* __restrict__ ref,
    const int* __restrict__ deformed,
    int* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
);

template void run_ccn_ring_buffer_row<float, float>(
    const float* __restrict__ ref,
    const float* __restrict__ deformed,
    float* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
);

template void run_ccn_ring_buffer_row<double, double>(
    const double* __restrict__ ref,
    const double* __restrict__ deformed,
    double* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
);

}
