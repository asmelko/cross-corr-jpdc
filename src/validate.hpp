#pragma once

#include <optional>
#include <algorithm>
#include <string>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <cmath>

#include "helpers.cuh"
#include "matrix.hpp"

namespace accs = boost::accumulators;


using namespace std::string_literals;

namespace cross {

class validation_results {
public:
    validation_results()
        :empty_(true), diff_mean_(0), diff_std_dev_(0)
    { }

    validation_results(double diff_mean, double diff_std_dev)
        :empty_(false), diff_mean_(diff_mean), diff_std_dev_(diff_std_dev)
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

template<typename MAT1, typename MAT2>
validation_results validate_result(const MAT1& result, const MAT2& valid_result) {
    // TODO: Throw when number of elements does not match

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

template<typename RESULTS, typename VALID>
validation_results compare_results(const RESULTS& results, const VALID& valid_results, bool is_fft) {
    if (is_fft) {
        auto norm = normalize_fft_results(results);
        return validate_result(norm, valid_results);
    }
    else {
        return validate_result(results, valid_results);
    }
}

inline dsize2_t result_matrix_size(dsize2_t ref_size, dsize2_t target_size) {
    return ref_size + target_size - 1;
}

inline void check_matrix_size(dsize2_t ref_size, dsize2_t target_size, dsize2_t result_size) {
    if (result_size != result_matrix_size(ref_size, target_size)) {
        throw std::runtime_error(
            "Invalid result matrix size "s +
            to_string(result_size) +
            ", expected "s +
            to_string(result_matrix_size(ref_size, target_size))
        );
    }
}

inline void check_num_result_matrices(dsize_t actual, dsize_t expected) {
    if (actual != expected) {
      throw std::runtime_error(
            "Invalid number of result matrices "s +
            std::to_string(actual) +
            ", expected "s +
            std::to_string(expected)
        );
    }
}

template<typename REF, typename TARGET, typename RESULT>
void cpu_cross_corr_one_to_one(const REF& ref, const TARGET& target, RESULT& result) {
    check_matrix_size(ref.matrix_size(), target.matrix_size(), result.matrix_size());
    check_num_result_matrices(result.num_matrices(), 1);

    naive_cpu_cross_corr(ref.view(), target.view(), result.view());
}

template<typename T, typename ALLOC>
data_array<T> cpu_cross_corr_one_to_one(const data_array<T, ALLOC>& ref, const data_array<T, ALLOC>& target) {
    data_array<T> result{result_matrix_size(ref.matrix_size(), target.matrix_size())};
    cpu_cross_corr_one_to_one(ref, target, result);
    return result;
}

template<typename REF, typename TARGET, typename RESULT>
void cpu_cross_corr_one_to_many(const REF& ref, const TARGET& target, RESULT& result) {
    check_matrix_size(ref.matrix_size(), target.matrix_size(), result.matrix_size());
    check_num_result_matrices(result.num_matrices(), target.num_matrices());

    for (auto i = 0; i < target.num_matrices(); ++i) {
        // TODO: Do in parallel
        naive_cpu_cross_corr(ref.view(), target.view(i), result.view(i));
    }
}

template<typename T, typename ALLOC>
data_array<T> cpu_cross_corr_one_to_many(const data_array<T, ALLOC>& ref, const data_array<T, ALLOC>& target) {
    data_array<T> result{result_matrix_size(ref.matrix_size(), target.matrix_size()), target.num_matrices()};
    cpu_cross_corr_one_to_many(ref, target, result);
    return result;
}

template<typename REF, typename TARGET, typename RESULT>
void cpu_cross_corr_n_to_mn(const REF& ref, const TARGET& target, RESULT& result) {
    check_matrix_size(ref.matrix_size(), target.matrix_size(), result.matrix_size());
    check_num_result_matrices(result.num_matrices(), target.num_matrices());

    if (target.num_matrices() % ref.num_matrices() != 0) {
        throw std::runtime_error(
            "Invalid ref and target data counts, "s +
            std::to_string(target.num_matrices()) +
            " is not divisible by "s +
            std::to_string(ref.num_matrices())
        );
    }

    // TODO: Do in parallel
    for (auto r = 0; r < ref.num_matrices(); ++r) {
        for (auto t = 0; t < target.num_matrices() / ref.num_matrices() ; ++t) {
            auto t_matrix_index = t * ref.num_matrices() + r;
            naive_cpu_cross_corr(ref.view(r), target.view(t_matrix_index), result.view(t_matrix_index));
        }
    }
}

template<typename T, typename ALLOC>
data_array<T> cpu_cross_corr_n_to_mn(const data_array<T, ALLOC>& ref, const data_array<T, ALLOC>& target) {
    data_array<T> result{result_matrix_size(ref.matrix_size(), target.matrix_size()), ref.num_matrices()};
    cpu_cross_corr_n_to_mn(ref, target, result);
    return result;
}

template<typename REF, typename TARGET, typename RESULT>
void cpu_cross_corr_n_to_m(const REF& ref, const TARGET& target, RESULT& result) {
    check_matrix_size(ref.matrix_size(), target.matrix_size(), result.matrix_size());
    check_num_result_matrices(result.num_matrices(), ref.num_matrices() * target.num_matrices());

    // TODO: Do in parallel
    for (auto r = 0; r < ref.num_matrices(); ++r) {
        for (auto t = 0; t < target.num_matrices(); ++t) {
            auto result_matrix_index = r * target.num_matrices() + t;
            naive_cpu_cross_corr(ref.view(r), target.view(t), result.view(result_matrix_index));
        }
    }
}

template<typename T, typename ALLOC>
data_array<T> cpu_cross_corr_n_to_m(const data_array<T, ALLOC>& ref, const data_array<T, ALLOC>& target) {
    data_array<T> result{result_matrix_size(ref.matrix_size(),target.matrix_size()), ref.num_matrices() * target.num_matrices()};
    cpu_cross_corr_n_to_m(ref, target, result);
    return result;
}

}