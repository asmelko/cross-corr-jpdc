#pragma once

#include <iostream>
#include <sstream>

#include <cufft.h>

namespace cross {


// Taken from original thesis src/cufft_helpers.hpp

#define FFTCH(status) cross::fft_check(status, __LINE__, __FILE__, #status)

static const char* cufft_get_error_message(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

inline void fft_check(cufftResult status, int line, const char* src_filename, const char* line_str = nullptr)
{
	if (status != CUFFT_SUCCESS)
	{
		std::stringstream ss;
		ss << "CUDA Error " << status << ":" << cufft_get_error_message(status) << " in " << src_filename << " (" << line << "):" << line_str << "\n";
		std::cerr << ss.str();
		throw std::runtime_error(ss.str());
	}
}

template<typename T>
inline cufftType fft_type_R2C();
template<>
inline cufftType fft_type_R2C<float>()
{
    return cufftType::CUFFT_R2C;
}
template<>
inline cufftType fft_type_R2C<double>()
{
    return cufftType::CUFFT_D2Z;
}


template<typename T>
inline cufftType fft_type_C2R();
template<>
inline cufftType fft_type_C2R<float>()
{
    return cufftType::CUFFT_C2R;
}
template<>
inline cufftType fft_type_C2R<double>()
{
    return cufftType::CUFFT_Z2D;
}

template<typename T>
struct real_trait
{
    using type = float;
};

template<>
struct real_trait<float>
{
    using type = cufftReal;
};
template<>
struct real_trait<double>
{
    using type = cufftDoubleReal;
};

template<typename T>
struct complex_trait
{
    using type = float;
};

template<>
struct complex_trait<cufftReal>
{
    using type = cufftComplex;
};
template<>
struct complex_trait<cufftDoubleReal>
{
    using type = cufftDoubleComplex;
};

template<typename T>
inline void fft_real_to_complex(cufftHandle plan, T* in, typename complex_trait<T>::type* out);

template<>
inline void fft_real_to_complex<cufftReal>(cufftHandle plan, cufftReal* in, complex_trait<cufftReal>::type* out)
{
    FFTCH(cufftExecR2C(plan, in, out));
}
template<>
inline void fft_real_to_complex<cufftDoubleReal>(cufftHandle plan, cufftDoubleReal* in, complex_trait<cufftDoubleReal>::type* out)
{
    FFTCH(cufftExecD2Z(plan, in, out));
}

template<typename T>
inline void fft_real_to_complex(cufftHandle plan, T* in_out);
template<>
inline void fft_real_to_complex<cufftReal>(cufftHandle plan, cufftReal* in_out)
{
    FFTCH(cufftExecR2C(plan, in_out, (cufftComplex*)in_out));
}
template<>
inline void fft_real_to_complex<cufftDoubleReal>(cufftHandle plan, cufftDoubleReal* in_out)
{
    FFTCH(cufftExecD2Z(plan, in_out, (cufftDoubleComplex*)in_out));
}

template<typename T>
inline void fft_complex_to_real(cufftHandle plan, typename complex_trait<T>::type* in, T* out);

template<>
inline void fft_complex_to_real<cufftReal>(cufftHandle plan, typename complex_trait<cufftReal>::type* in, float* out)
{
    FFTCH(cufftExecC2R(plan, in, out));
}
template<>
inline void fft_complex_to_real<cufftDoubleReal>(cufftHandle plan, typename complex_trait<cufftDoubleReal>::type* in, double* out)
{
    FFTCH(cufftExecZ2D(plan, in, out));
}

template<typename T>
inline void fft_complex_to_real(cufftHandle plan, T* in_out);
template<>
inline void fft_complex_to_real<cufftReal>(cufftHandle plan, cufftReal* in_out)
{
    FFTCH(cufftExecC2R(plan, (cufftComplex*)in_out, in_out));
}
template<>
inline void fft_complex_to_real<cufftDoubleReal>(cufftHandle plan, cufftDoubleReal* in_out)
{
    FFTCH(cufftExecZ2D(plan, (cufftDoubleComplex*)in_out, in_out));
}

}