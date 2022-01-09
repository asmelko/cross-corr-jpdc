#pragma once

#include "cuda.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "types.cuh"

namespace cg = cooperative_groups;

namespace cross {

template <typename T>
__device__ T* shared_memory_proxy()
{
    // double2 to ensure 16B alignment
	extern __shared__ double2 memory[];
	return reinterpret_cast<T*>(memory);
}

template<typename T>
class shared_mem_buffer{
public:

    using value_type = T;
    using size_type = dsize_t;
    using reference = value_type&;
    using const_reference = const value_type&;

    __device__ shared_mem_buffer(T* data, dsize_t size)
        :data_(data), size_(size)
    {

    }

    __device__ static shared_mem_buffer<T> allocate(T** shared, dsize_t size) {
        T* data = *shared;
        *shared += size;
        return shared_mem_buffer<T>(data, size);
    }

    __device__ reference operator [](dsize_t i) {
        return data_[i];
    }

    __device__ const_reference operator [](dsize_t i) const {
        return data_[i];
    }

    __device__ size_type size() const {
        return size_;
    }

    // __device__ size_type load(cg::thread_block ctb, row_slice<T>&& slice, dsize_t size, dsize_t offset = 0) {
    //     // TODO: Asserts
    //     size_type copy_size = min(size, slice.size());
    //     auto data = data_ + offset;
    //     for (size_type i = ctb.thread_index().x; i < copy_size; i += ctb.size()) {
    //         data[i] = slice[i];
    //     }
    //     return copy_size;
    // }

    __device__ size_type load(cg::thread_block ctb, const T* src, dsize_t size, dsize_t offset = 0) {
        // TODO: Asserts
        size_type copy_size = min(size, size_ - offset);
        auto data = data_ + offset;
        for (size_type i = ctb.thread_index().x; i < copy_size; i += ctb.size()) {
            data[i] = src[i];
        }
        return copy_size;
    }

    __device__ value_type* data() const {
        return data_;
    }
private:
    value_type* data_;
    dsize_t size_;
};


}