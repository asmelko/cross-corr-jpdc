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
class one_to_one: public cross_corr_alg<T, ALLOC> {
public:
    one_to_one(bool is_fft, std::size_t num_measurements)
        :cross_corr_alg<T, ALLOC>(is_fft, num_measurements)
    {}

    validation_results validate_impl(const std::optional<std::filesystem::path>& valid_data) const override {
        if (valid_data.has_value()) {
            auto val = validate_with_precomputed(
                load_matrix_from_csv<T, no_padding>(*valid_data)
            );
            return val.validate(this->results(), this->is_fft());
        } else {
            auto val = validate_with_computed_one_to_one(this->refs(), this->targets());
            return val.validate(this->results(), this->is_fft());
        }
        return validation_results{};
    }
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_original_alg_one_to_one: public one_to_one<T, ALLOC> {
public:
    naive_original_alg_one_to_one()
        :one_to_one<T, ALLOC>(false, labels.size()), ref_(), target_(), result_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return target_;
    }

    const data_array<T, ALLOC>& results() const override {
        return result_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(def_path);
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_cross_corr_naive_original(
                d_ref_,
                d_target_,
                d_result_,
                target_.matrix_size(),
                result_.matrix_size(),
                1,
                1
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_original_alg_one_to_one<T, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};

template<typename T, dsize_t THREADS_PER_BLOCK, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_ring_buffer_row_alg: public one_to_one<T, ALLOC> {
public:
    naive_ring_buffer_row_alg()
        :one_to_one<T, ALLOC> (false, labels.size()), ref_(), target_(), res_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return target_;
    }

    const data_array<T, ALLOC>& results() const override {
        return res_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);
        res_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_res_, res_.size());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_ccn_ring_buffer_row(
                d_ref_,
                d_target_,
                d_res_,
                target_.matrix_size(),
                res_.matrix_size(),
                THREADS_PER_BLOCK
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(res_, d_res_);
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> res_;

    T* d_ref_;
    T* d_target_;
    T* d_res_;
};

template<typename T, dsize_t THREADS_PER_BLOCK, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_ring_buffer_row_alg<T, THREADS_PER_BLOCK, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class fft_original_alg: public one_to_one<T, ALLOC> {
public:
    fft_original_alg()
        :one_to_one<T, ALLOC>(true, labels.size()), ref_(), target_(), res_(), fft_buffer_size_(0)
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return target_;
    }

    const data_array<T, ALLOC>& results() const override {
        return res_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {

        ref_ = load_matrix_from_csv<T, relative_zero_padding<2>, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, relative_zero_padding<2>, ALLOC>(target_path);

        if (DEBUG)
        {
            std::ofstream out_file("../data/ref.csv");
            ref_.store_to_csv(out_file);

            out_file = std::ofstream("../data/target.csv");
            target_.store_to_csv(out_file);
        }


        res_ = data_array<T, ALLOC>{ref_.matrix_size()};
        fft_buffer_size_ = ref_.matrix_size().y * (ref_.matrix_size().x / 2 + 1);

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_res_, res_.size());

        cuda_malloc(&d_ref_fft_, fft_buffer_size_);
        cuda_malloc(&d_target_fft_, fft_buffer_size_);

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);


        FFTCH(cufftPlan2d(&fft_plan_, ref_.matrix_size().y, ref_.matrix_size().x, fft_type_R2C<T>()));
        FFTCH(cufftPlan2d(&fft_inv_plan_, ref_.matrix_size().y, ref_.matrix_size().x, fft_type_C2R<T>()));
    }

    void run_impl() override {
        CPU_MEASURE(0,
            fft_real_to_complex(fft_plan_, d_ref_, d_ref_fft_);
            fft_real_to_complex(fft_plan_, d_target_, d_target_fft_);
        );

        if (DEBUG)
        {
            std::vector<fft_complex_t> tmp(fft_buffer_size_);
            cuda_memcpy_from_device(tmp.data(), d_ref_fft_, tmp.size());

            std::ofstream out("../data/ref_fft.csv");
            out << tmp << std::endl;

            cuda_memcpy_from_device(tmp.data(), d_target_fft_, tmp.size());

            out = std::ofstream("../data/target_fft.csv");
            out << tmp << std::endl;
        }


        CUDA_MEASURE(1,
            run_hadamard_original(
                d_ref_fft_,
                d_target_fft_,
                {ref_.matrix_size().y, (ref_.matrix_size().x / 2) + 1},
                1,
                1,
                256)
        );

        if (DEBUG)
        {
            CUCH(cudaDeviceSynchronize());
            std::vector<fft_complex_t> tmp(fft_buffer_size_);
            cuda_memcpy_from_device(tmp.data(), d_target_fft_, tmp.size());

            std::ofstream out("../data/hadamard.csv");
            out << tmp << std::endl;
        }

        CUDA_MEASURE(2,
            fft_complex_to_real(fft_inv_plan_, d_target_fft_, d_res_)
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(res_, d_res_);
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> res_;

    T* d_ref_;
    T* d_target_;
    T* d_res_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize_t fft_buffer_size_;

    fft_complex_t* d_ref_fft_;
    fft_complex_t* d_target_fft_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> fft_original_alg<T, DEBUG, ALLOC>::labels{
    "Total",
    "Forward FFT",
    "Hadamard",
    "Inverse FFT"
};


}