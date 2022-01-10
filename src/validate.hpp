#pragma once

#include <optional>
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

class validation_results {
public:
    validation_results()
        :empty_(true), diff_mean_(0), diff_std_dev_(0)
    { }

    validation_results(double diff_mean, double diff_std_dev)
        :empty_(true), diff_mean_(diff_mean), diff_std_dev_(diff_std_dev)
    { }

    double get_diff_mean() const {
        return diff_mean_;
    }

    double get_diff_std_dev() const {
        return diff_std_dev_;
    }

    bool empty() const {
        return empty_;
    }
private:
    bool empty_;
    double diff_mean_;
    double diff_std_dev_;
};


std::ostream& operator <<(std::ostream& out, const validation_results& res) {
    if (res.empty()) {
        out << "No validation" << "\n";
    } else {
        out << "Difference from valid values:" << "\n";
        out << "Mean: " << res.get_diff_mean() << "\n";
        out << "Stddev: " << res.get_diff_std_dev() << "\n";
    }
    return out;
}

template<typename MAT>
static typename MAT::value_type hadamard_and_sum(const MAT& ref, const MAT& target, int offset_x, int offset_y) {
    // Part of the reference matrix overlapping the target matrix
    dsize_t x_ref_start = (dsize_t)std::max(offset_x, 0);
    dsize_t x_ref_end = (dsize_t)std::min((int)ref.size().x + offset_x, (int)ref.size().x);
    dsize_t y_ref_start = (dsize_t)std::max(offset_y, 0);
    dsize_t y_ref_end = (dsize_t)std::min((int)ref.size().y + offset_y, (int)ref.size().y);

    auto sum = typename MAT::value_type{};
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


template<typename MAT1, typename MAT2, typename MAT3>
void naive_cpu_cross_corr(const MAT1& ref, const MAT2& target, MAT3&& res) {
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

template<typename MAT, typename RES = typename MAT::value_type>
data_single<RES> naive_cpu_cross_corr(const MAT& ref, const MAT& target, dsize2_t search_size) {

    data_single<RES> res{search_size};
    naive_cpu_cross_corr(ref, target, res.view());
    return res;
}



template<typename MAT1, typename MAT2>
validation_results validate_result(const MAT1& result, const MAT2& valid_result) {
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

    return validation_results{
        accs::mean(acc),
        std::sqrt(accs::variance(acc))
    };

}



template<typename DATA>
class validator {
public:


    validator(DATA&& valid_results)
        :valid_results_(std::move(valid_results))
    {

    }

    template<typename RESULTS>
    validation_results validate(const RESULTS& results, bool is_fft) {
        if (is_fft) {
            auto norm = normalize_fft_results(results);
            return validate_result(norm, valid_results_);
        }
        else {
            return validate_result(results, valid_results_);
        }
    }
protected:
    DATA valid_results_;
};

template<typename T, typename ALLOC>
validator<data_single<T>> validate_with_computed_one_to_one(const data_single<T, ALLOC>& ref, const data_single<T, ALLOC>& target) {
    return validator<data_single<T>>(naive_cpu_cross_corr<matrix_view<const T>, T>(ref.view(), target.view(), ref.matrix_size() + target.matrix_size() - 1));
}

template<typename T, typename ALLOC>
validator<data_array<T>> validate_with_computed_one_to_many(const data_single<T, ALLOC>& ref, const data_array<T, ALLOC>& target) {
    data_array<T> valid_results{target.num_matrices(), ref.matrix_size() + target.matrix_size() - 1};
    for (auto i = 0; i < target.num_matrices(); ++i) {
        // TODO: Do in parallel
        naive_cpu_cross_corr(ref.view(), target.view(i), valid_results.view(i));
    }
    return validator<data_array<T>>(std::move(valid_results));
}

template<typename DATA>
validator<DATA> validate_with_precomputed(DATA&& valid_results) {
    return validator<DATA>{std::move(valid_results)};
}

}