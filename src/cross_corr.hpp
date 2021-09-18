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

template<typename MAT>
class cross_corr_alg {
public:
    cross_corr_alg(bool is_fft, std::size_t num_measurements)
        :is_fft_(is_fft), sw_(num_measurements)
    {}

    void prepare(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) {
        start_ = sw_.now();
        prepare_impl(ref_path, def_path);
    }

    void run() {
        run_impl();
    }

    void finalize() {
        finalize_impl();
        sw_.cpu_manual_measure(0, start_);
    }

    virtual const MAT& results() const = 0;
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

    virtual void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) = 0;
    virtual void run_impl() = 0;
    virtual void finalize_impl() = 0;

    template<typename PADDING>
    static MAT load_matrix_from_csv(const std::filesystem::path& path) {
        std::ifstream file(path);
        return MAT::template load_from_csv<PADDING>(file);
    }

    sw_clock::time_point start_;
};

template<typename MAT, bool DEBUG = false>
class naive_original_alg: public cross_corr_alg<MAT> {
public:
    naive_original_alg()
        :cross_corr_alg<MAT>(false, labels.size()), ref_(), target_(), res_()
    {

    }

    const MAT& results() const override {
        return res_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = cross_corr_alg<MAT>::template load_matrix_from_csv<no_padding>(ref_path);
        target_ = cross_corr_alg<MAT>::template load_matrix_from_csv<no_padding>(target_path);
        res_ = MAT{ref_.size() + target_.size() - 1};

        cuda_malloc(&d_ref_, ref_.area());
        cuda_malloc(&d_target_, target_.area());
        cuda_malloc(&d_res_, res_.area());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_cross_corr_naive_original(
                d_ref_,
                d_target_,
                d_res_,
                target_.size(),
                res_.size(),
                1,
                1
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

    MAT ref_;
    MAT target_;

    MAT res_;

    typename MAT::value_type* d_ref_;
    typename MAT::value_type* d_target_;
    typename MAT::value_type* d_res_;
};

template<typename MAT, bool DEBUG>
std::vector<std::string> naive_original_alg<MAT, DEBUG>::labels{
    "Total",
    "Kernel"
};

template<typename MAT, dsize_t THREADS_PER_BLOCK, bool DEBUG = false>
class naive_ring_buffer_row_alg: public cross_corr_alg<MAT> {
public:
    naive_ring_buffer_row_alg()
        :cross_corr_alg<MAT>(false, labels.size()), ref_(), target_(), res_()
    {

    }

    const MAT& results() const override {
        return res_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = cross_corr_alg<MAT>::template load_matrix_from_csv<no_padding>(ref_path);
        target_ = cross_corr_alg<MAT>::template load_matrix_from_csv<no_padding>(target_path);
        res_ = MAT{ref_.size() + target_.size() - 1};

        cuda_malloc(&d_ref_, ref_.area());
        cuda_malloc(&d_target_, target_.area());
        cuda_malloc(&d_res_, res_.area());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_ccn_ring_buffer_row(
                d_ref_,
                d_target_,
                d_res_,
                target_.size(),
                res_.size(),
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

    MAT ref_;
    MAT target_;

    MAT res_;

    typename MAT::value_type* d_ref_;
    typename MAT::value_type* d_target_;
    typename MAT::value_type* d_res_;
};

template<typename MAT, dsize_t THREADS_PER_BLOCK, bool DEBUG>
std::vector<std::string> naive_ring_buffer_row_alg<MAT, THREADS_PER_BLOCK, DEBUG>::labels{
    "Total",
    "Kernel"
};


template<typename MAT, bool DEBUG = false>
class fft_original_alg: public cross_corr_alg<MAT> {
public:
    fft_original_alg()
        :cross_corr_alg<MAT>(true, labels.size()), ref_(), target_(), res_(), fft_buffer_size_(0)
    {

    }

    const MAT& results() const override {
        return res_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {

        ref_ = cross_corr_alg<MAT>::template load_matrix_from_csv<relative_zero_padding<2>>(ref_path);
        target_ = cross_corr_alg<MAT>::template load_matrix_from_csv<relative_zero_padding<2>>(target_path);

        if (DEBUG)
        {
            std::ofstream out_file("../data/ref.csv");
            ref_.store_to_csv(out_file);

            out_file = std::ofstream("../data/target.csv");
            target_.store_to_csv(out_file);
        }


        res_ = MAT{ref_.size()};
        fft_buffer_size_ = ref_.size().y * (ref_.size().x / 2 + 1);

        cuda_malloc(&d_ref_, ref_.area());
        cuda_malloc(&d_target_, target_.area());
        cuda_malloc(&d_res_, res_.area());

        cuda_malloc(&d_ref_fft_, fft_buffer_size_);
        cuda_malloc(&d_target_fft_, fft_buffer_size_);

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);


        FFTCH(cufftPlan2d(&fft_plan_, ref_.size().y, ref_.size().x, fft_type_R2C<typename MAT::value_type>()));
        FFTCH(cufftPlan2d(&fft_inv_plan_, ref_.size().y, ref_.size().x, fft_type_C2R<typename MAT::value_type>()));
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
                {ref_.size().y, (ref_.size().x / 2) + 1},
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
    using fft_real_t = typename real_trait<typename MAT::value_type>::type;
    using fft_complex_t = typename complex_trait<typename MAT::value_type>::type;

    static std::vector<std::string> labels;

    MAT ref_;
    MAT target_;

    MAT res_;

    typename MAT::value_type* d_ref_;
    typename MAT::value_type* d_target_;
    typename MAT::value_type* d_res_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize_t fft_buffer_size_;

    fft_complex_t* d_ref_fft_;
    fft_complex_t* d_target_fft_;
};

template<typename MAT, bool DEBUG>
std::vector<std::string> fft_original_alg<MAT, DEBUG>::labels{
    "Total",
    "Forward FFT",
    "Hadamard",
    "Inverse FFT"
};

}