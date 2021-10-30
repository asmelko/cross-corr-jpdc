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
#include "one_to_one.hpp"
#include "one_to_many.hpp"

using namespace cross;

using data_type = float;

template<typename T>
data_single<T> compute_valid_results(
    const std::filesystem::path& in_ref,
    const std::filesystem::path& in_tgt
) {

    std::ifstream ref_file(in_ref);
    std::ifstream tgt_file(in_tgt);
    auto ref = data_single<T>::template load_from_csv<no_padding>(ref_file);
    auto tgt = data_single<T>::template load_from_csv<no_padding>(tgt_file);

    return naive_cpu_cross_corr(ref, tgt, ref.size() + tgt.size() - 1);
}

template<typename T>
data_single<T> get_valid_result_one_to_one(
    const std::filesystem::path& in_ref,
    const std::filesystem::path& in_tgt,
    std::optional<std::filesystem::path> valid_result
) {
    if (valid_result) {
        std::ifstream valid_res_file(*valid_result);
        return data_single<T>::template load_from_csv<no_padding>(valid_res_file);
    }
    else {
        std::ifstream ref_file(in_ref);
        std::ifstream tgt_file(in_tgt);
        auto ref = data_single<T>::template load_from_csv<no_padding>(ref_file);
        auto tgt = data_single<T>::template load_from_csv<no_padding>(tgt_file);

        return naive_cpu_cross_corr(ref.view(), tgt.view(), ref.matrix_size() + tgt.matrix_size() - 1);
    }
}

template<typename T>
data_array<T> get_valid_result_one_to_many(
    const std::filesystem::path& in_ref,
    const std::filesystem::path& in_tgt,
    std::optional<std::filesystem::path> valid_result
) {
    if (valid_result) {
        std::ifstream valid_res_file(*valid_result);
        return data_array<T>::template load_from_csv<no_padding>(valid_res_file);
    }
    else {
        std::ifstream ref_file(in_ref);
        std::ifstream tgt_file(in_tgt);
        auto ref = data_single<T>::template load_from_csv<no_padding>(ref_file);
        auto tgt = data_array<T>::template load_from_csv<no_padding>(tgt_file);
        data_array<T> result{tgt.num_matrices(), ref.matrix_size() + tgt.matrix_size() - 1};

        for (auto i = 0; i < tgt.num_matrices(); ++i) {
            // TODO: Do in parallel
            naive_cpu_cross_corr(ref.view(), tgt.view(i), result.view(i));
        }

        return result;
    }
}

template<typename ALG, typename DATA>
void validate_results(
    const ALG& alg,
    const DATA& valid_result
){
    auto res = alg.results();

    results stats;
    if (alg.is_fft()) {
        auto norm = normalize_fft_results(res);
        stats = validate_result(norm, valid_result);
    }
    else {
        stats = validate_result(res, valid_result);
    }

    std::cout << "Difference from valid values:" << "\n";
    std::cout << "Mean: " << stats.diff_mean << "\n";
    std::cout << "Stddev: " << stats.diff_std_dev << "\n";
}

template<typename ALG>
void run_one_to_many(
    const std::filesystem::path& ref_path,
    const std::filesystem::path& def_path,
    const std::filesystem::path& out_path,
    const std::filesystem::path& measurements_path,
    std::optional<std::filesystem::path> valid_results
) {
    ALG alg;
    std::cerr << "Loading inputs\n";
    alg.prepare(ref_path, def_path);

    std::cerr << "Running test alg\n";
    alg.run();

    std::cerr << "Copying output data to host\n";
    alg.finalize();

    auto res = alg.results();
    std::ofstream out_file(out_path);
    res.store_to_csv(out_file);

    std::cerr << "Validating results\n";
    validate_results(alg, get_valid_result_one_to_many<typename ALG::data_type>(ref_path, def_path, valid_results));


    std::ofstream measurements_file(measurements_path);
    auto labels = alg.measurement_labels();
    auto measurements = alg.measurements();
    to_csv(measurements_file, labels);
    to_csv<std::chrono::nanoseconds>(measurements_file, measurements);
}


template<typename ALG>
void run_one_to_one(
    const std::filesystem::path& ref_path,
    const std::filesystem::path& def_path,
    const std::filesystem::path& out_path,
    const std::filesystem::path& measurements_path,
    std::optional<std::filesystem::path> valid_results
) {
    ALG alg;
    std::cerr << "Loading inputs\n";
    alg.prepare(ref_path, def_path);

    std::cerr << "Running test alg\n";
    alg.run();

    std::cerr << "Copying output data to host\n";
    alg.finalize();

    auto res = alg.results();
    std::ofstream out_file(out_path);
    res.store_to_csv(out_file);

    std::cerr << "Validating results\n";
    validate_results(alg, get_valid_result_one_to_one<typename ALG::data_type>(ref_path, def_path, valid_results));


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
    const std::filesystem::path&,
    std::optional<std::filesystem::path>
)>> algorithms{
    {"nai_orig", run_one_to_one<naive_original_alg<data_type, false, pinned_allocator<data_type>>>},
    {"nai_rows_128", run_one_to_one<naive_ring_buffer_row_alg<data_type, 128, false, pinned_allocator<data_type>>>},
    {"nai_rows_256", run_one_to_one<naive_ring_buffer_row_alg<data_type, 256, false, pinned_allocator<data_type>>>},
    {"nai_def_block_128", run_one_to_many<naive_def_per_block<data_type, 128, false, pinned_allocator<data_type>>>},
    {"nai_def_block_256", run_one_to_many<naive_def_per_block<data_type, 256, false, pinned_allocator<data_type>>>},
    {"fft_orig", run_one_to_one<fft_original_alg<data_type, false, pinned_allocator<data_type>>>}
};

int main(int argc, char **argv) {
    // TODO: Better argument parsing
    if (argc < 6 || argc > 7) {
        std::cerr << "Invalid number of arguments, expected between " << 6 << " and " << 7 << " , got " << argc - 1 << "\n";
        std::cerr << "Usage: " << argv[0] << "<alg> <ref_path> <target_path> <out_path> <measurements_path> [valid_results]\n";
        return 1;
    }
    try {
        auto fnc = algorithms.find(argv[1]);
        if (fnc == algorithms.end()) {
            // TODO: List of available algorithms
            std::cerr << "Unknown algorithm \"" << argv[1] << "\", expected one of " << std::endl;
            return 1;
        }

        std::optional<std::filesystem::path> valid_results{};
        if (argc == 7) {
            valid_results = argv[6];
        }

        fnc->second(argv[2], argv[3], argv[4], argv[5], valid_results);
        return 0;
    }
    catch (std::exception& e) {
        std::cerr << "Exception occured: " << e.what() << std::endl;
        return 2;
    }
}