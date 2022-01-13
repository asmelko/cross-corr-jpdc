#pragma once

#include <vector>
#include <chrono>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"

#include "kernels.cuh"

namespace cross {

template<typename T, typename ALLOC = std::allocator<T>>
class one_to_many: public cross_corr_alg<T, ALLOC> {
public:
    one_to_many(bool is_fft, std::size_t num_measurements)
        :cross_corr_alg<T,ALLOC>(is_fft, num_measurements)
    {}

    data_array<T> get_valid_results() const override {
        return cpu_cross_corr_one_to_many(this->refs(), this->targets());
    }
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class cpu_one_to_many: public one_to_many<T, ALLOC> {
public:
    cpu_one_to_many()
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
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
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        auto result_matrix_size = ref_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

    }

    void run_impl() override {
        cpu_cross_corr_one_to_many(ref_, targets_, results_);
    }

    void finalize_impl() override {
    }


private:
    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> cpu_one_to_many<T, DEBUG, ALLOC>::labels{
    "Total",
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_original_alg_one_to_many: public one_to_many<T, ALLOC> {
public:
    naive_original_alg_one_to_many()
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
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
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);
        auto result_matrix_size = ref_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_cross_corr_naive_original(
                d_ref_,
                d_targets_,
                d_results_,
                targets_.matrix_size(),
                results_.matrix_size(),
                1,
                targets_.num_matrices()
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

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_original_alg_one_to_many<T, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};

template<typename T, dsize_t THREADS_PER_BLOCK, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_def_per_block: public one_to_many<T, ALLOC> {
public:
    naive_def_per_block()
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
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
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);
        auto result_matrix_size = ref_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {

        // TODO: Number of blocks argument
        auto num_blocks = 0;

        CUDA_MEASURE(1,
            run_ccn_def_per_block(
                d_ref_,
                d_targets_,
                d_results_,
                ref_.matrix_size(),
                results_.matrix_size(),
                targets_.num_matrices(),
                num_blocks,
                THREADS_PER_BLOCK
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

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;
};

template<typename T, dsize_t THREADS_PER_BLOCK, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_def_per_block<T, THREADS_PER_BLOCK, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};

}