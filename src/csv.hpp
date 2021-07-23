#pragma once

#include <iostream>
#include <vector>
#include <chrono>

#include "matrix.hpp"

namespace cross {

template<typename T>
void to_csv(std::ostream& out, T value) {
    out << value << "\n";
}

template<typename T>
void to_csv(std::ostream& out, const std::vector<T>& values) {
    auto sep = "";
    for (auto&& val: values) {
        out << sep << val;
        sep = ",";
    }
    out << "\n";
}

template<typename OUT_DURATION, typename REP, typename PERIOD>
void to_csv(std::ostream& out, const std::vector<std::chrono::duration<REP, PERIOD>>& durations) {
    auto sep = "";
    for (auto&& dur: durations) {
        out << sep << std::chrono::duration_cast<OUT_DURATION>(dur).count();
        sep = ",";
    }
    out << "\n";
}

template<typename T, typename ALLOC>
void to_csv(std::ostream& out, const matrix<T, ALLOC>& matrix) {
    matrix.store_to_csv(out);
}

}