#pragma once

#include <string>
#include <exception>
#include <sstream>
#include <iostream>

#include <cuda_runtime.h>

#include "types.cuh"


namespace cross {

#define CUCH(status) cross::cuda_check(status, __LINE__, __FILE__, #status)

inline void cuda_check(cudaError_t status, int line, const char* src_filename, const char* line_str = nullptr)
{
	if (status != cudaSuccess)
	{
		std::stringstream ss;
		ss << "CUDA Error " << status << ":" << cudaGetErrorString(status) << " in " << src_filename << " (" << line << "):" << line_str << "\n";
		std::cerr << ss.str();
		throw std::runtime_error(ss.str());
	}
}

// Divides two integers and rounds the result up
template<typename T, typename U>
inline __host__ __device__ T div_up(T a, U b)
{
	return (a + b - 1) / b;
}

// TODO: Static assert that the generic function is not called
template<typename T>
class parser {
public:
	static inline T from_string(const std::string& in) {
		throw std::runtime_error{"Invalid parser, should not happen"};
	}
};


template<>
class parser<float> {
public:
	static float from_string(const std::string& in){
		return std::stof(in);
	}
};

template<>
class parser<double> {
	static double from_string(const std::string& in) {
		return std::stod(in);
	}
};

/** Allocates device buffer large enough to hold \p num instances of T
 *
 * This helper prevents common error of forgetting the sizeof(T)
 * when allocating buffers
 */
template<typename T>
void cuda_malloc(T** p, dsize_t num) {
	CUCH(cudaMalloc(p, num * sizeof(T)));
}

template<typename T>
void cuda_memcpy_to_device(T* dst, T* src, dsize_t num) {
	CUCH(cudaMemcpy(dst, src, num * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename MAT>
void cuda_memcpy_to_device(typename MAT::value_type* dst, MAT& src) {
	cuda_memcpy_to_device(dst, src.data(), src.area());
}


template<typename T>
void cuda_memcpy_from_device(T* dst, T* src, dsize_t num) {
	CUCH(cudaMemcpy(dst, src, num * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename MAT>
void cuda_memcpy_from_device(MAT& dst, typename MAT::value_type* src) {
	cuda_memcpy_from_device(dst.data(), src, dst.area());
}

}
