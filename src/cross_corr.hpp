#pragma once

#include <vector>
#include <chrono>

#include <cufft.h>

#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"

#include "kernels.cuh"

#include "fft_helpers.hpp"

namespace cross {

using sw_clock = std::chrono::high_resolution_clock;

template<typename T, typename ALLOC = std::allocator<T>>
class cross_corr_alg {
public:
    using data_type = T;

    cross_corr_alg(bool is_fft, std::size_t num_measurements)
        :is_fft_(is_fft), sw_(num_measurements)
    {}

    void run() {
        run_impl();
    }

    void finalize() {
        finalize_impl();
        sw_.cpu_manual_measure(0, start_);
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
    StopWatch<sw_clock> sw_;

    virtual void run_impl() = 0;
    virtual void finalize_impl() = 0;

    template<typename PADDING>
    static data_single<T, ALLOC> load_matrix_from_csv_single(const std::filesystem::path& path) {
        std::ifstream file(path);
        return data_single<T, ALLOC>::template load_from_csv<PADDING>(file);
    }

    /**
     * Load matrix array from many csv files, one matrix per csv file
     */
    template<typename PADDING>
    static data_array<T,ALLOC> load_matrix_array_from_csv(const std::vector<std::filesystem::path>& paths) {
        std::vector<std::ifstream> inputs(paths.size());
        for (auto&& path: paths) {
            inputs.push_back(std::ifstream{path});
        }
        return data_array<T,ALLOC>::template load_from_csv<PADDING>(std::move(inputs));
    }

    /**
     * Load matrix array from single csv file containing multiple matrices
     */
    template<typename PADDING>
    static data_array<T,ALLOC> load_matrix_array_from_csv(const std::filesystem::path& path) {
        std::ifstream file(path);
        return data_array<T,ALLOC>::template load_from_csv<PADDING>(file);
    }

    void start_timer() {
        start_ = sw_.now();
    }

    sw_clock::time_point start_;
};


}