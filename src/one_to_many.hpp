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

    void prepare(const std::filesystem::path& ref_path, const std::vector<std::filesystem::path>& def_paths) {
        this->start_timer();
        prepare_impl(ref_path, def_paths);
    }

    void prepare(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) {
        this->start_timer();
        prepare_impl(ref_path, def_path);
    }

    validation_results validate_impl(const std::optional<std::filesystem::path>& valid_data) override {
        if (valid_data.has_value()) {
            auto val = validate_with_precomputed(
                load_matrix_array_from_csv<T, no_padding>(*valid_data)
            );
            return val.validate(results(), this->is_fft());
        } else {
            auto val = validate_with_computed_one_to_many(get_ref(), get_targets());
            return val.validate(results(), this->is_fft());
        }
        return validation_results{};
    }

    virtual const data_array<T, ALLOC>& results() const = 0;
protected:
    virtual void prepare_impl(const std::filesystem::path& ref_path, const std::vector<std::filesystem::path>& def_paths) = 0;
    virtual void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) = 0;

    virtual const data_single<T, ALLOC>& get_ref() const = 0;
    virtual const data_array<T, ALLOC>& get_targets() const = 0;
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_original_alg: public one_to_many<T, ALLOC> {
public:
    naive_original_alg()
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::vector<std::filesystem::path>& def_paths) override {
        prepare_impl_common(
            load_matrix_from_csv_single<T, no_padding, ALLOC>(ref_path),
            load_matrix_array_from_csv<T, no_padding, ALLOC>(def_paths)
        );
    }

    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        prepare_impl_common(
            load_matrix_from_csv_single<T, no_padding, ALLOC>(ref_path),
            load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path)
        );
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


    const data_single<T, ALLOC>& get_ref() const override {
        return ref_;
    }
    const data_array<T, ALLOC>& get_targets() const {
        return targets_;
    }
private:

    static std::vector<std::string> labels;

    data_single<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;

    void prepare_impl_common(data_single<T, ALLOC> &&ref, data_array<T, ALLOC> &&targets) {
        ref_ = std::move(ref);
        targets_ = std::move(targets);
        auto result_matrix_size = ref_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{targets_.num_matrices(), result_matrix_size};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_original_alg<T, DEBUG, ALLOC>::labels{
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

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::vector<std::filesystem::path>& def_paths) override {
        prepare_impl_common(
            load_matrix_from_csv_single<T, no_padding, ALLOC>(ref_path),
            load_matrix_array_from_csv<T, no_padding, ALLOC>(def_paths)
        );
    }

    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        prepare_impl_common(
            load_matrix_from_csv_single<T, no_padding, ALLOC>(ref_path),
            load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path)
        );
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

    const data_single<T, ALLOC>& get_ref() const override {
        return ref_;
    }
    const data_array<T, ALLOC>& get_targets() const {
        return targets_;
    }
private:

    static std::vector<std::string> labels;

    data_single<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;

    void prepare_impl_common(data_single<T, ALLOC> &&ref, data_array<T, ALLOC> &&targets) {
        ref_ = std::move(ref);
        targets_ = std::move(targets);
        auto result_matrix_size = ref_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{targets_.num_matrices(), result_matrix_size};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }
};

template<typename T, dsize_t THREADS_PER_BLOCK, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_def_per_block<T, THREADS_PER_BLOCK, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};

}