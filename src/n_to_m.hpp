#pragma once

#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"

#include "kernels.cuh"

using namespace std::string_literals;

namespace cross {

template<typename T, typename ALLOC = std::allocator<T>>
class n_to_m: public cross_corr_alg<T, ALLOC> {
public:
    n_to_m(bool is_fft, std::size_t num_measurements)
        :cross_corr_alg<T,ALLOC>(is_fft, num_measurements)
    {}

    data_array<T> get_valid_results() const override {
        return cpu_cross_corr_n_to_m(this->refs(), this->targets());
    }
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class cpu_n_to_m: public n_to_m<T, ALLOC> {
public:
    cpu_n_to_m()
        :n_to_m<T, ALLOC>(false, labels.size()), refs_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }
    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, refs_.num_matrices() * targets_.num_matrices()};
    }

    void run_impl() override {
        cpu_cross_corr_n_to_m(refs_, targets_, results_);
    }

    void finalize_impl() override {
    }


private:
    static std::vector<std::string> labels;

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> cpu_n_to_m<T, DEBUG, ALLOC>::labels{
    "Total",
};

}
