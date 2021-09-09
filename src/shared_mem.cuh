#pragma once

#include "cuda.h"

namespace cross {

template <typename T>
__device__ T* shared_memory_proxy()
{
    // double2 to ensure 16B alignment
	extern __shared__ double2 memory[];
	return reinterpret_cast<T*>(memory);
}

}