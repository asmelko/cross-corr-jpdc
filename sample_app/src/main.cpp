#include <iostream>
#include <string>
#include <filesystem>
#include <exception>

#include <cuda_runtime.h>

#include "cuda_helpers.cuh"
#include "kernels.cuh"
#include "matrix.hpp"

using namespace cross;

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <left_input_matrix_path> <right_input_matrices_path>" << std::endl;
        return 1;
    }

    try {
        auto left_input = load_matrix_from_csv<float, no_padding>(std::filesystem::path{argv[1]});
        auto right_inputs = load_matrix_array_from_csv<float, no_padding>(std::filesystem::path{argv[2]});

        if (left_input.matrix_size() != right_inputs.matrix_size()) {
            std::cerr << "Currently the implementation can only compute cross-correlation for left and right matrices of the same size." << std::endl;
            return 2;
        }

        auto result_matrix_size = left_input.matrix_size() + right_inputs.matrix_size() - 1;
        auto results = data_array<float>{result_matrix_size, right_inputs.num_matrices()};

        float* d_left_input;
        float* d_right_inputs;
        float* d_results;

        cuda_malloc(&d_left_input, left_input.size());
        cuda_malloc(&d_right_inputs, right_inputs.size());
        cuda_malloc(&d_results, results.size());

        cuda_memcpy_to_device(d_left_input, left_input);
        cuda_memcpy_to_device(d_right_inputs, right_inputs);

        const dsize_t warps_per_thread_block = 8;
        const dsize_t shifts_per_thread_right_matrix = 2;
        const dsize_t right_matrices_per_thread = 2;
        const dsize_t left_rows_per_iteration = 2;

        run_ccn_shuffle_one_to_many_multirow_both_multimat_right(
            d_left_input,
            d_right_inputs,
            d_results,
            left_input.matrix_size(),
            results.matrix_size(),
            right_inputs.num_matrices(),
            warps_per_thread_block,
            shifts_per_thread_right_matrix,
            right_matrices_per_thread,
            left_rows_per_iteration
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());

        cuda_memcpy_from_device(results, d_results);

        cuda_free(d_results);
        cuda_free(d_right_inputs);
        cuda_free(d_left_input);

        results.store_to_csv(std::cout);
    } catch(std::exception& e) {
        std::cerr << "Exception occured: " << e.what() << std::endl;
        return 2;
    }

}