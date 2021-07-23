#pragma once


#include <cufft.h>

#include "matrix.hpp"
#include "helpers.cuh"

#include "kernels.cuh"

#include "fft_helpers.hpp"

namespace cross {

template<typename MAT>
class cross_corr_alg {
public:
    virtual void prepare(const std::string& ref_path, const std::string&) = 0;
    virtual void run() = 0;
    virtual void finalize() = 0;

    virtual const MAT& results() = 0;
protected:
    template<typename PADDING>
    static MAT load_matrix_from_csv(const std::string& path) {
        std::ifstream file(path);
        return MAT::template load_from_csv<PADDING>(file);
    }
};

template<typename MAT, bool DEBUG = false>
class naive_original_alg: public cross_corr_alg<MAT> {
public:
    naive_original_alg()
        :ref_(), target_(), res_()
    {

    }

    void prepare(const std::string& ref_path, const std::string& target_path) override {
        ref_ = cross_corr_alg<MAT>::template load_matrix_from_csv<no_padding>(ref_path);
        target_ = cross_corr_alg<MAT>::template load_matrix_from_csv<no_padding>(target_path);
        res_ = MAT{ref_.size() + target_.size() - 1};

        cuda_malloc(&d_ref_, ref_.area());
        cuda_malloc(&d_target_, target_.area());
        cuda_malloc(&d_res_, res_.area());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run() override {
        run_cross_corr_naive_original<typename MAT::value_type, typename MAT::value_type>(
            d_ref_,
            d_target_,
            d_res_,
            target_.size(),
            res_.size(),
            1,
            1
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize() override {
        cuda_memcpy_from_device(res_, d_res_);
    }

    const MAT& results() const override {
        return res_;
    }

private:
    MAT ref_;
    MAT target_;

    MAT res_;

    typename MAT::value_type* d_ref_;
    typename MAT::value_type* d_target_;
    typename MAT::value_type* d_res_;
};

template<typename MAT, bool DEBUG = false>
class fft_original_alg: public cross_corr_alg<MAT> {
public:
    fft_original_alg()
        :ref_(), target_(), res_(), fft_buffer_size_(0)
    {

    }

    void prepare(const std::string& ref_path, const std::string& target_path) {

        ref_ = cross_corr_alg<MAT>::template load_matrix_from_csv<relative_zero_padding<2>>(ref_path);
        target_ = cross_corr_alg<MAT>::template load_matrix_from_csv<relative_zero_padding<2>>(target_path);

        if (DEBUG)
        {
            std::ofstream out_file("../data/ref.csv");
            ref_.store_to_csv(out_file);

            std::ofstream out_file("../data/target.csv");
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

    void run() {
        fft_real_to_complex(fft_plan_, d_ref_, d_ref_fft_);
        fft_real_to_complex(fft_plan_, d_target_, d_target_fft_);

        if (DEBUG)
        {
            std::vector<fft_complex_t> tmp(fft_buffer_size_);
            cuda_memcpy_from_device(tmp.data(), d_ref_fft_, tmp.size());

            std::ofstream out("../data/ref_fft.csv");
            out << tmp << std::endl;
        }

        if (DEBUG)
        {
            std::vector<fft_complex_t> tmp(fft_buffer_size_);
            cuda_memcpy_from_device(tmp.data(), d_target_fft_, tmp.size());

            std::ofstream out("../data/target_fft.csv");
            out << tmp << std::endl;
        }

        run_hadamard_original(
            d_ref_fft_,
            d_target_fft_,
            {ref_.size().y, (ref_.size().x / 2) + 1},
            1,
            1,
            256);

        if (DEBUG)
        {
            CUCH(cudaDeviceSynchronize());
            std::vector<fft_complex_t> tmp(fft_buffer_size_);
            cuda_memcpy_from_device(tmp.data(), d_target_fft_, tmp.size());

            std::ofstream out("../data/hadamard.csv");
            out << tmp << std::endl;
        }

        fft_complex_to_real(fft_inv_plan_, d_target_fft_, d_res_);

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize() {
        cuda_memcpy_from_device(res_, d_res_);
    }

    const MAT& results() {
        return res_;
    }

private:
    using fft_real_t = typename real_trait<typename MAT::value_type>::type;
    using fft_complex_t = typename complex_trait<typename MAT::value_type>::type;

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

}