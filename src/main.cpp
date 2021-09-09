#include <iostream>
#include <fstream>
#include <filesystem>

#include "validate.hpp"
#include "matrix.hpp"
#include "cross_corr.hpp"
#include "allocator.cuh"
#include "csv.hpp"
#include "fft_helpers.hpp"

using namespace cross;

using mat = matrix<float, pinned_allocator<float>>;
using cpu_mat = matrix<float>;

cpu_mat get_valid_results(
    const std::filesystem::path& in_ref,
    const std::filesystem::path& in_tgt,
    const std::filesystem::path& res_out
) {
    // TODO: Allow user to specify path to the valid outputs
    // if (true) {
    //     const std::filesystem::path valid_path{"../data/out_valid_1024.csv"};
    //     std::ifstream valid_file(valid_path);
    //     return cpu_mat::load_from_csv<no_padding>(valid_file);
    // }
    // else {
        std::ifstream ref_file(in_ref);
        std::ifstream tgt_file(in_tgt);
        auto ref = cpu_mat::load_from_csv<no_padding>(ref_file);
        auto tgt = cpu_mat::load_from_csv<no_padding>(tgt_file);

        auto valid_res = naive_cpu_cross_corr(ref, tgt, ref.size() + tgt.size() - 1);
        std::ofstream valid_out(res_out.parent_path() / res_out.filename().replace_extension(std::string{".valid"} + res_out.extension().generic_string()));
        valid_res.store_to_csv(valid_out);

        return valid_res;
    // }
}

template<typename ALG>
void validate_results(
    const std::filesystem::path& in_ref,
    const std::filesystem::path& in_tgt,
    const std::filesystem::path& res_out,
    const ALG& alg
){


    auto valid_res = get_valid_results(in_ref, in_tgt, res_out);

    auto res = alg.results();

    results stats;
    if (alg.is_fft()) {
        auto norm = normalize_fft_results<mat, cpu_mat>(res);
        std::ofstream norm_out(res_out.parent_path() / res_out.filename().replace_extension(std::string{".norm"} + res_out.extension().generic_string()));
        norm.store_to_csv(norm_out);
        stats = validate_result(norm, valid_res);
    }
    else {
        stats = validate_result(res, valid_res);
    }

    std::cout << "Percentage difference from valid values:" << "\n";
    std::cout << "Mean: " << stats.diff_mean << "%\n";
    std::cout << "Stddev: " << stats.diff_std_dev << "%\n";

    std::ofstream out_file(res_out);
    res.store_to_csv(out_file);
}

int main(int argc, char **argv) {
    // TODO: Better argument parsing
    if (argc != 5) {
        std::cerr << "Invalid number of arguments, expected " << 5 << " , got " << argc - 1 << "\n";
        std::cerr << "Usage: " << argv[0] << " <ref_path> <target_path> <out_path> <measurements_path>\n";
        return 1;
    }
    try {
        //naive_original_alg<mat> alg;
        //fft_original_alg<mat, true> alg;
        naive_ring_buffer_row_alg<mat, 256, true> alg;

        std::cerr << "Loading inputs\n";
        alg.prepare(argv[1], argv[2]);

        std::cerr << "Running test alg\n";
        alg.run();

        std::cerr << "Copying output data to host\n";
        alg.finalize();

        // DEBUG
        // auto res = alg.results();
        // std::ofstream out_file(argv[3]);
        // res.store_to_csv(out_file);

        std::cerr << "Validating results\n";
        validate_results(argv[1], argv[2], argv[3], alg);


        std::ofstream measurements_file(argv[4]);
        auto labels = alg.measurement_labels();
        auto measurements = alg.measurements();
        to_csv(measurements_file, labels);
        to_csv<std::chrono::nanoseconds>(measurements_file, measurements);
    }
    catch (std::exception& e) {
        std::cerr << "Exception occured: " << e.what() << std::endl;
        return 2;
    }
}