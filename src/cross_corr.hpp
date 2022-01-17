#pragma once

#include <vector>
#include <chrono>
#include <filesystem>

#include <cufft.h>

#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"

#include "kernels.cuh"

#include "fft_helpers.hpp"

namespace cross {

using sw_clock = std::chrono::high_resolution_clock;



/**
 * Load matrix array from many csv files, one matrix per csv file
 */
template<typename T, typename PADDING, typename ALLOC = std::allocator<T>>
std::tuple<data_array<T,ALLOC>, std::vector<dsize_t>> load_matrix_array_from_multiple_csv(const std::vector<std::filesystem::path>& paths) {
    std::vector<std::ifstream> inputs(paths.size());
    for (auto&& path: paths) {
        inputs.push_back(std::ifstream{path});
    }
    return data_array<T,ALLOC>::template load_from_csv<PADDING>(std::move(inputs));
}

/**
 * Load matrix array from single csv file containing multiple matrices
 */
template<typename T, typename PADDING, typename ALLOC = std::allocator<T>>
data_array<T,ALLOC> load_matrix_array_from_csv(const std::filesystem::path& path) {
    std::ifstream file(path);
    return data_array<T,ALLOC>::template load_from_csv<PADDING>(file);
}

/**
 * Load matrix from a csv file
 */
template<typename T, typename PADDING, typename ALLOC = std::allocator<T>>
data_array<T,ALLOC> load_matrix_from_csv(const std::filesystem::path& path) {
    // TODO: Maybe throw if more than one matrix
    return load_matrix_array_from_csv<T, PADDING, ALLOC>(path);
}

template<typename T, typename ALLOC = std::allocator<T>>
class cross_corr_alg {
public:
    using data_type = T;

    cross_corr_alg(bool is_fft, std::size_t num_measurements)
        :is_fft_(is_fft), sw_(num_measurements)
    {}

    void prepare(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) {
        this->start_timer();
        prepare_impl(ref_path, def_path);
    }

    void run() {
        run_impl();
    }

    void finalize() {
        finalize_impl();
        sw_.cpu_measure(0, start_);
    }

    virtual const data_array<T, ALLOC>& refs() const = 0;

    virtual const data_array<T, ALLOC>& targets() const = 0;

    virtual const data_array<T, ALLOC>& results() const = 0;


    validation_results validate(const std::optional<std::filesystem::path>& valid_data_path = std::nullopt) const {
        auto valid = valid_results(valid_data_path);
        if (this->is_fft()) {
            return validate_result(normalize_fft_results(this->results()), valid);
        } else {
            return validate_result(this->results(), valid);
        }
    }


    virtual const std::vector<std::string>& measurement_labels() const = 0;

    bool is_fft() const {
        return is_fft_;
    }

    const std::vector<sw_clock::duration> measurements() const {
        return sw_.results();
    }
protected:

    bool is_fft_;
    stopwatch<sw_clock> sw_;

    virtual void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) = 0;
    virtual void run_impl() = 0;
    virtual void finalize_impl() = 0;
    virtual data_array<T> get_valid_results() const = 0;

    void start_timer() {
        start_ = sw_.now();
    }

    sw_clock::time_point start_;

private:
    data_array<T> valid_results(const std::optional<std::filesystem::path>& valid_data_path = std::nullopt) const {
        if (valid_data_path.has_value()) {
            return load_matrix_array_from_csv<T, no_padding>(*valid_data_path);
        } else {
            return get_valid_results();
        }

    }
};


}