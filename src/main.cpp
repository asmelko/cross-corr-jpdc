#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <unordered_map>

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

    std::cout << "Difference from valid values:" << "\n";
    std::cout << "Mean: " << stats.diff_mean << "\n";
    std::cout << "Stddev: " << stats.diff_std_dev << "\n";

    std::ofstream out_file(res_out);
    res.store_to_csv(out_file);
}

template<typename ALG>
void run_measurement(
    const std::filesystem::path& ref_path,
    const std::filesystem::path& def_path,
    const std::filesystem::path& out_path,
    const std::filesystem::path& measurements_path
) {
    ALG alg;
    std::cerr << "Loading inputs\n";
    alg.prepare(ref_path, def_path);

    std::cerr << "Running test alg\n";
    alg.run();

    std::cerr << "Copying output data to host\n";
    alg.finalize();

    // DEBUG
    auto res = alg.results();
    std::ofstream out_file(out_path);
    res.store_to_csv(out_file);

    std::cerr << "Validating results\n";
    validate_results(ref_path, def_path, out_path, alg);


    std::ofstream measurements_file(measurements_path);
    auto labels = alg.measurement_labels();
    auto measurements = alg.measurements();
    to_csv(measurements_file, labels);
    to_csv<std::chrono::nanoseconds>(measurements_file, measurements);
}

static std::unordered_map<std::string, std::function<void(
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&
)>> algorithms{
    {"nai_orig", run_measurement<naive_original_alg<mat>>},
    {"nai_rows_128", run_measurement<naive_ring_buffer_row_alg<mat, 128>>},
    {"nai_rows_256", run_measurement<naive_ring_buffer_row_alg<mat, 256>>},
    {"fft_orig", run_measurement<fft_original_alg<mat>>}
};

int main(int argc, char **argv) {
    // TODO: Better argument parsing
    if (argc != 6) {
        std::cerr << "Invalid number of arguments, expected " << 6 << " , got " << argc - 1 << "\n";
        std::cerr << "Usage: " << argv[0] << "<alg> <ref_path> <target_path> <out_path> <measurements_path>\n";
        return 1;
    }
    try {
        auto fnc = algorithms.find(argv[1]);
        if (fnc == algorithms.end()) {
            // TODO: List of available algorithms
            std::cerr << "Unknown algorithm \"" << argv[1] << "\", expected one of " << std::endl;
            return 1;
        }

        fnc->second(argv[2], argv[3], argv[4], argv[5]);
        return 0;
    }
    catch (std::exception& e) {
        std::cerr << "Exception occured: " << e.what() << std::endl;
        return 2;
    }
}