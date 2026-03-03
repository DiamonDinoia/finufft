#ifndef CUFINUFFT_TYPES_H
#define CUFINUFFT_TYPES_H

#include <cuda/std/array>
#include <cufft.h>
#include <cufinufft/defs.h>
#include <cufinufft_opts.h>
#include <finufft_common/spread_opts.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <type_traits>

#include <cuComplex.h>

using CUFINUFFT_BIGINT = int;

// Marco Barbone 8/5/2024, replaced the ugly trick with std::conditional
// to define cuda_complex
// by using std::conditional and std::is_same, we can define cuda_complex
// if T is float, cuda_complex<T> is cuFloatComplex
// if T is double, cuda_complex<T> is cuDoubleComplex
// where cuFloatComplex and cuDoubleComplex are defined in cuComplex.h
// TODO: migrate to cuda/std/complex and remove this
//       Issue: cufft seems not to support cuda::std::complex
//       A reinterpret_cast should be enough
template<typename T>
using cuda_complex = typename std::conditional<
    std::is_same<T, float>::value, cuFloatComplex,
    typename std::conditional<std::is_same<T, double>::value, cuDoubleComplex,
                              void>::type>::type;

template<typename T> inline T *dethrust(thrust::device_ptr<T> ptr) {
  return thrust::raw_pointer_cast(ptr);
}
template<typename T> inline thrust::device_ptr<T> enthrust(T *ptr) {
  return thrust::device_pointer_cast(ptr);
}

class DeviceSwitcher {
private:
  int orig_device;

  static int get_orig_device() noexcept {
    int device{};
    cudaGetDevice(&device);
    return device;
  }

public:
  explicit DeviceSwitcher(int newDevice) : orig_device{get_orig_device()} {
    if (cudaSetDevice(newDevice) != cudaSuccess)
        throw int(FINUFFT_ERR_CUDA_FAILURE);
  }

  ~DeviceSwitcher() {
    if (cudaSetDevice(orig_device) != cudaSuccess) {
      std::cerr << "failure reverting to original CUDA device; Exiting..." << std::endl;
      std::terminate();
    }
  }
};

template<typename T>
struct ThrustAllocatorAsync : public thrust::device_malloc_allocator<T> {
public:
  using Base      = thrust::device_malloc_allocator<T>;
  using pointer   = typename Base::pointer;
  using size_type = typename Base::size_type;

private:
  cudaStream_t stream;
  int deviceID;
  bool pool;

public:
  // Prefer explicit stream; no default ctor needed if you always pass alloc to
  // device_vector
  explicit ThrustAllocatorAsync(cudaStream_t s, int ID, bool supports_pools)
      : stream(s), deviceID(ID), pool(supports_pools) {}

  pointer allocate(size_type n) {
    DeviceSwitcher switcher(deviceID);
    T *p = nullptr;
    auto err =
        pool ? cudaMallocAsync(&p, n * sizeof(T), stream) : cudaMalloc(&p, n * sizeof(T));
    if (err != cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
    return enthrust(p);
  }

  void deallocate(pointer p, size_type) {
    DeviceSwitcher switcher(deviceID);
    auto err = pool ? cudaFreeAsync(dethrust(p), stream) : cudaFree(dethrust(p));
    if (err != cudaSuccess) { // something went really, really wrong, memory is corrupted
      std::cerr << "error while deallocating GPU memory! Exiting ..." << std::endl;
      std::terminate();
    }
  }
};

template<typename T> using gpu_array = thrust::device_vector<T, ThrustAllocatorAsync<T>>;

template<typename T> inline T *dethrust(gpu_array<T> &arr) {
  return thrust::raw_pointer_cast(arr.data());
}
template<typename T> inline const T *dethrust(const gpu_array<T> &arr) {
  return thrust::raw_pointer_cast(arr.data());
}
template<typename T>
inline cuda::std::array<T *, 3> dethrust(cuda::std::array<gpu_array<T>, 3> &arr) {
  cuda::std::array<T *, 3> res;
  for (int i = 0; i < 3; ++i) res[i] = dethrust(arr[i]);
  return res;
}
template<typename T>
inline cuda::std::array<const T *, 3> dethrust(
    const cuda::std::array<gpu_array<T>, 3> &arr) {
  cuda::std::array<const T *, 3> res;
  for (int i = 0; i < 3; ++i) res[i] = dethrust(arr[i]);
  return res;
}

template<typename T> struct cufinufft_plan_t {
  // FIXME: we want to make data members private in the future.
  // Not yet possible at the moment, since not all functions working
  // on plans have been converted to members.

  cufinufft_opts opts;
  bool supports_pools = false;
  finufft_spread_opts spopts;

  ThrustAllocatorAsync<int> ialloc{(cudaStream_t)opts.gpu_stream, opts.gpu_device_id,
                                   supports_pools};
  ThrustAllocatorAsync<T> alloc{(cudaStream_t)opts.gpu_stream, opts.gpu_device_id,
                                supports_pools};
  ThrustAllocatorAsync<cuda_complex<T>> calloc{(cudaStream_t)opts.gpu_stream,
                                               opts.gpu_device_id, supports_pools};

  int type                                    = 0;
  int dim                                     = 0;
  CUFINUFFT_BIGINT M                          = 0;
  cuda::std::array<CUFINUFFT_BIGINT, 3> nf123 = {0, 0, 0};
  cuda::std::array<CUFINUFFT_BIGINT, 3> mstu  = {0, 0, 0};
  int ntransf                                 = 0;
  int batchsize                               = 0;
  int iflag                                   = 0;

  int totalnumsubprob                        = 0;
  cuda::std::array<gpu_array<T>, 3> fwkerhalf = {
      gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}};

  // for type 1,2 it is a pointer to kx, ky, kz (no new allocs), for type 3 it
  // for t3: allocated as "primed" (scaled) src pts x'_j, etc
  cuda::std::array<const T *, 3> kxyz    = {nullptr, nullptr, nullptr};
  cuda::std::array<gpu_array<T>, 3> kxyzp = {gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc},
                                            gpu_array<T>{0, alloc}};
  gpu_array<cuda_complex<T>> CpBatch{0, calloc}; // working array of prephased strengths

  // no allocs here
  cuda_complex<T> *c = nullptr;
  gpu_array<cuda_complex<T>> fwp{0, calloc};
  cuda_complex<T> *fw = nullptr;
  cuda_complex<T> *fk = nullptr;

  // Type 3 specific
  struct {
    cuda::std::array<T, 3> X = {0, 0, 0}, C = {0, 0, 0}, S = {0, 0, 0}, D = {0, 0, 0},
                           h = {0, 0, 0}, gam = {0, 0, 0};
  } type3_params;
  int N                                 = 0; // number of NU freq pts (type 3 only)
  CUFINUFFT_BIGINT nf                   = 0;
  cuda::std::array<const T *, 3> STU    = {nullptr, nullptr, nullptr};
  cuda::std::array<gpu_array<T>, 3> STUp = {gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc},
                                           gpu_array<T>{0, alloc}};
  T tol                                 = 0;
  // inner type 2 plan for type 3
  cufinufft_plan_t<T> *t2_plan = nullptr;

  gpu_array<cuda_complex<T>> prephase{0, calloc}; // pre-phase, for all input NU pts
  gpu_array<cuda_complex<T>> deconv{0, calloc};   // reciprocal of kernel FT, phase, all
                                                 // output NU pts

  // Arrays that used in subprob method
  gpu_array<int> idxnupts{0, ialloc};   // length: #nupts, index of the nupts in the
                                       // bin-sorted order
  gpu_array<int> sortidx{0, ialloc};    // length: #nupts, order inside the bin the nupt
                                       // belongs to
  gpu_array<int> numsubprob{0, ialloc}; // length: #bins,  number of subproblems in each
                                       // bin
  gpu_array<int> binsize{0, ialloc}; // length: #bins, number of nonuniform ponits in each
                                    // bin
  gpu_array<int> binstartpts{0, ialloc}; // length: #bins, exclusive scan of array binsize
  gpu_array<int> subprob_to_bin{0, ialloc}; // length: #subproblems, the bin the subproblem
                                           // works on
  gpu_array<int> subprobstartpts{0, ialloc}; // length: #bins, exclusive scan of array
                                            // numsubprob

  // Arrays for 3d (need to sort out)
  gpu_array<int> numnupts{0, ialloc};
  gpu_array<int> subprob_to_nupts{0, ialloc};

  cufftHandle fftplan = 0;
  cudaStream_t stream = 0;

  bool eps_too_small = false;

  cufinufft_plan_t() = delete;
  cufinufft_plan_t(int type_, int dim_, const int *nmodes, int iflag_, int ntransf_,
                   T tol_, const cufinufft_opts &opts_);
  cufinufft_plan_t &operator=(cufinufft_plan_t &) = delete;

  ~cufinufft_plan_t() {
    DeviceSwitcher switcher(opts.gpu_device_id);
    if (fftplan) cufftDestroy(fftplan);
    delete t2_plan;
  }

private:
  void alloc1d();
  void alloc2d();
  void alloc3d();

  void alloc1d_nupts();
  void alloc2d_nupts();
  void alloc3d_nupts();

  void exec1(cuda_complex<T> *d_c, cuda_complex<T> *d_fk);
  void exec2(cuda_complex<T> *d_c, cuda_complex<T> *d_fk);
  void exec3(cuda_complex<T> *d_c, cuda_complex<T> *d_fk);

  void deconvolve(int blksize) const;
  template<int modeord, int ndim> void deconvolve_nd(int blksize) const;

  void setpts_12(int M_, const T *d_kx, const T *d_ky, const T *d_kz);
  void allocate();
  void allocate_nupts();

public:
  void setpts(int M_, const T *d_kx, const T *d_ky, const T *d_kz, int N_, const T *d_s,
              const T *d_t, const T *d_u);
  // FIXME: we want to make this "const" in the future
  void exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk);
};

template<typename T> static inline constexpr cufftType_t cufft_type();
template<> inline constexpr cufftType_t cufft_type<float>() { return CUFFT_C2C; }

template<> inline constexpr cufftType_t cufft_type<double>() { return CUFFT_Z2Z; }

static inline cufftResult cufft_ex(cufftHandle plan, cufftComplex *idata,
                                   cufftComplex *odata, int direction) {
  return cufftExecC2C(plan, idata, odata, direction);
}
static inline cufftResult cufft_ex(cufftHandle plan, cufftDoubleComplex *idata,
                                   cufftDoubleComplex *odata, int direction) {
  return cufftExecZ2Z(plan, idata, odata, direction);
}

#endif
