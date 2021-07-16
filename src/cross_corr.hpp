#pragma once

#include "matrix.hpp"
#include "helpers.cuh"

#include "kernels.cuh"

namespace cross {

template<typename MAT>
class naive {
public:
    naive(MAT&& ref, MAT&& target)
        :ref_(std::move(ref)), target_(std::move(target)),
        res_((ref_.size() + target_.size() - 1))
    {

    }

    void prepare() {
        cuda_malloc(&d_ref_, ref_.area());
        cuda_malloc(&d_target_, target_.area());
        cuda_malloc(&d_res_, res_.area());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run() {
        run_cross_corr_naive_original<typename MAT::value_type, typename MAT::value_type>(
            d_target_,
            d_ref_,
            d_res_,
            target_.size(),
            res_.size(),
            1,
            1
        );

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
    MAT ref_;
    MAT target_;

    MAT res_;

    typename MAT::value_type* d_ref_;
    typename MAT::value_type* d_target_;
    typename MAT::value_type* d_res_;
};

}