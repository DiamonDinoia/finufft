#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <complex>
#include <cstdint>

#include <cuComplex.h>
#include <cufinufft/types.h>

#include <cuda_runtime.h>

#include <sys/time.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 || defined(__clang__)
#else
__inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

namespace cufinufft {
namespace utils {
class WithCudaDevice {
public:
  WithCudaDevice(int device) {
    cudaGetDevice(&orig_device_);
    cudaSetDevice(device);
  }

  ~WithCudaDevice() { cudaSetDevice(orig_device_); }

private:
  int orig_device_;
};

// jfm timer class
class CNTime {
public:
  void start();
  double restart();
  double elapsedsec();

private:
  struct timeval initial;
};

// ahb math helpers
CUFINUFFT_BIGINT next235beven(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT b);

template<typename T> T infnorm(int n, std::complex<T> *a) {
  T nrm = 0.0;
  for (int m = 0; m < n; ++m) {
    T aa = real(conj(a[m]) * a[m]);
    if (aa > nrm) nrm = aa;
  }
  return sqrt(nrm);
}

#ifdef __CUDA_ARCH__
__forceinline__ __device__ auto interval(const int ns, const float x) {
  const auto xstart = __float2int_ru(__fmaf_ru(ns, -.5f, x));
  const auto xend   = __float2int_rd(__fmaf_rd(ns, .5f, x));
  return int2{xstart, xend};
}
__forceinline__ __device__ auto interval(const int ns, const double x) {
  const auto xstart = __double2int_ru(__fma_ru(ns, -.5, x));
  const auto xend   = __double2int_rd(__fma_rd(ns, .5, x));
  return int2{xstart, xend};
}
#endif

// Define a macro to check if NVCC version is >= 11.3
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
#if (__CUDACC_VER_MAJOR__ > 11) || \
    (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 3 && __CUDA_ARCH__ >= 600)
#define ALLOCA_SUPPORTED 1
#else
#define ALLOCA_SUPPORTED 0
#endif
#else
#define ALLOCA_SUPPORTED 0
#endif

#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 900
#define COMPUTE_CAPABILITY_90_OR_HIGHER 1
#else
#define COMPUTE_CAPABILITY_90_OR_HIGHER 0
#endif
#else
#define COMPUTE_CAPABILITY_90_OR_HIGHER 0
#endif

} // namespace utils
} // namespace cufinufft

#endif
