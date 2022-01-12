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
class n_to_mn: public cross_corr_alg<T, ALLOC> {
public:
    n_to_mn(bool is_fft, std::size_t num_measurements)
        :cross_corr_alg<T,ALLOC>(is_fft, num_measurements)
    {}

    validation_results validate_impl(const std::optional<std::filesystem::path>& valid_data) const override {
        if (valid_data.has_value()) {
            auto val = validate_with_precomputed(
                load_matrix_array_from_csv<T, no_padding>(*valid_data)
            );
            return val.validate(this->results(), this->is_fft());
        } else {
            auto val = validate_with_computed_n_to_mn(this->refs(), this->targets());
            return val.validate(this->results(), this->is_fft());
        }
        return validation_results{};
    }
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_original_alg_n_to_mn: public n_to_mn<T, ALLOC> {
public:
    naive_original_alg_n_to_mn()
        :n_to_mn<T, ALLOC>(false, labels.size()), refs_(), targets_(), results_()
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

        if (targets_.num_matrices() % refs_.num_matrices() != 0) {
            throw std::runtime_error(
                "Invalid input data counts, "s +
                std::to_string(targets_.num_matrices()) +
                " is not divisible by "s +
                std::to_string(refs_.num_matrices())
            );
        }

        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

        cuda_malloc(&d_refs_, refs_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());

        cuda_memcpy_to_device(d_refs_, refs_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_cross_corr_naive_original(
                d_refs_,
                d_targets_,
                d_results_,
                targets_.matrix_size(),
                results_.matrix_size(),
                refs_.num_matrices(),
                targets_.num_matrices() / refs_.num_matrices()
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_refs_;
    T* d_targets_;
    T* d_results_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_original_alg_n_to_mn<T, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};


}