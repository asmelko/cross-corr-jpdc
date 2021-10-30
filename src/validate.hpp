#pragma once

#include <algorithm>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <cmath>

#include "helpers.cuh"
#include "matrix.hpp"

namespace accs = boost::accumulators;

namespace cross {

struct results {
    double diff_mean;
    double diff_std_dev;
};


template<typename MAT>
static typename MAT::value_type hadamard_and_sum(const MAT& ref, const MAT& target, int offset_x, int offset_y) {
    // Part of the reference matrix overlapping the target matrix
    dsize_t x_ref_start = (dsize_t)std::max(offset_x, 0);
    dsize_t x_ref_end = (dsize_t)std::min((int)ref.size().x + offset_x, (int)ref.size().x);
    dsize_t y_ref_start = (dsize_t)std::max(offset_y, 0);
    dsize_t y_ref_end = (dsize_t)std::min((int)ref.size().y + offset_y, (int)ref.size().y);

    typename MAT::value_type sum = 0;
    for (dsize_t y_ref = y_ref_start; y_ref < y_ref_end; ++y_ref) {
        for (dsize_t x_ref = x_ref_start; x_ref < x_ref_end; ++x_ref) {
            // Corresponding part of the target matrix
            dsize_t x_shifted = x_ref - offset_x;
            dsize_t y_shifted = y_ref - offset_y;

            sum += target[dsize2_t{x_shifted, y_shifted}] * ref[dsize2_t{x_ref, y_ref}];
        }
    }

    return sum;
}


template<typename MAT>
void naive_cpu_cross_corr(const MAT& ref, const MAT& target, MAT&& res) {
    auto search_size = res.size();

    // TODO: Why is there a -1?
    dsize2_t half_size = (search_size - 1) / 2;

    // For all possible shifts
    for (int y = -(int)half_size.y; y <= (int)half_size.y; ++y) {
        for (int x = -(int)half_size.x; x <= (int)half_size.x; ++x) {
            dsize_t res_offset = (dsize2_t{x + half_size.x, y + half_size.y}).linear_idx(search_size.x);
            // Compute element sum of hadamard product of overlapping parts of the matrix
            res.data()[res_offset] = hadamard_and_sum(ref, target, x, y);
        }
    }
}

template<typename MAT>
data_single<typename MAT::value_type> naive_cpu_cross_corr(const MAT& ref, const MAT& target, dsize2_t search_size) {

    data_single<typename MAT::value_type> res{search_size};
    naive_cpu_cross_corr(ref, target, res.view());
    return res;
}



template<typename MAT1, typename MAT2>
results validate_result(const MAT1& result, const MAT2& valid_result) {
    std::vector<double> differences;

    std::transform(
        std::begin(result),
        std::end(result),
        std::begin(valid_result),
        std::back_inserter(differences),
        [](typename MAT1::value_type a, typename MAT2::value_type b){
            return a - b;
        });

    accs::accumulator_set<
        double,
        accs::stats<
            accs::tag::mean,
            accs::tag::variance(accs::lazy)
        >
    > acc;

    std::for_each(std::begin(differences), std::end(differences), std::bind<void>(std::ref(acc), std::placeholders::_1));

    return results{
        accs::mean(acc),
        std::sqrt(accs::variance(acc))
    };

}

}