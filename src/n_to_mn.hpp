#pragma once

#include <vector>
#include <chrono>
#include <filesystem>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"

#include "kernels.cuh"

namespace cross {

template<typename T, typename ALLOC = std::allocator<T>>
class n_to_mn: public cross_corr_alg<T, ALLOC> {
public:
    n_to_mn(bool is_fft, std::size_t num_measurements)
        :cross_corr_alg<T,ALLOC>(is_fft, num_measurements)
    {}

    void prepare(const std::vector<std::filesystem::path>& ref_paths, const std::vector<std::filesystem::path>& def_paths) {
        this->start_timer();
        prepare_impl(ref_path, def_paths);
    }

    void prepare(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) {
        this->start_timer();
        prepare_impl(ref_path, def_path);
    }

    virtual const data_array<T, ALLOC>& results() const = 0;
protected:
    virtual data_array<T, ALLOC> load_refs() = 0;
    virtual data_array<T, ALLOC> load_defs() = 0;
    virtual void prepare_impl() = 0;
};

}