#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <iostream>

#include <cufinufft/common.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>

using namespace cufinufft::common;

#include "spreadinterp1d.cuh"

namespace cufinufft {
namespace spreadinterp {

// Functor to handle function selection (nuptsdriven vs subprob)
struct Interp1DDispatcher {
  template<int ns, typename T>
  int operator()(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) const {
    switch (d_plan->opts.gpu_method) {
    case 1:
      return cuinterp1d_nuptsdriven<T, ns>(nf1, M, d_plan, blksize);
    case 2:
      return cuinterp1d_subprob<T, ns>(nf1, M, d_plan, blksize);
    case 3:
      return cuinterp1d_output_driven<T, ns>(nf1, M, d_plan, blksize);
    default:
      std::cerr << "[cuinterp1d] error: incorrect method, should be 1, 2 or 3\n";
      return FINUFFT_ERR_METHOD_NOTVALID;
    }
  }
};

// Updated cuinterp1d using generic dispatch
template<typename T> int cuinterp1d(cufinufft_plan_t<T> *d_plan, int blksize) {
  /*
   A wrapper for different interpolation methods.

   Methods available:
      (1) Non-uniform points driven
      (2) Subproblem
      (3) Output driven

   Melody Shih 11/21/21

   Now the function is updated to dispatch based on ns. This is to avoid alloca which
   it seems slower according to the MRI community.
   Marco Barbone 01/30/25
  */
  return launch_dispatch_ns<Interp1DDispatcher, T>(Interp1DDispatcher(),
                                                   d_plan->spopts.nspread, d_plan->nf1,
                                                   d_plan->M, d_plan, blksize);
}

template<typename T, int ns>
int cuinterp1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream = d_plan->stream;
  dim3 threadsPerBlock;
  dim3 blocks;

  T es_c          = d_plan->spopts.ES_c;
  T es_beta       = d_plan->spopts.ES_beta;
  T sigma         = d_plan->opts.upsampfac;
  int *d_idxnupts = d_plan->idxnupts;

  T *d_kx               = d_plan->kx;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  threadsPerBlock.x =
      std::min(optimal_block_threads(d_plan->opts.gpu_device_id), (unsigned)M);
  threadsPerBlock.y = 1;
  blocks.x          = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y          = 1;

  if (d_plan->opts.gpu_kerevalmeth) {
    for (int t = 0; t < blksize; t++) {
      interp_1d_nuptsdriven<T, 1, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      interp_1d_nuptsdriven<T, 0, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template<typename T, int ns>
int cuinterp1d_subprob(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream       = d_plan->stream;
  T es_c             = d_plan->spopts.ES_c;
  T es_beta          = d_plan->spopts.ES_beta;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int numbins    = ceil((T)nf1 / bin_size_x);

  T *d_kx               = d_plan->kx;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  T sigma = d_plan->opts.upsampfac;

  const auto sharedplanorysize = shared_memory_required<T>(
      1, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);

  if (d_plan->opts.gpu_kerevalmeth) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(interp_1d_subprob<T, 1, ns>, 1, *d_plan) != 0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      interp_1d_subprob<T, 1, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_binstartpts,
          d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(interp_1d_subprob<T, 0, ns>, 1, *d_plan) != 0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      interp_1d_subprob<T, 0, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_binstartpts,
          d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template<typename T, int ns>
int cuinterp1d_output_driven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream       = d_plan->stream;
  T es_c             = d_plan->spopts.ES_c;
  T es_beta          = d_plan->spopts.ES_beta;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int numbins    = ceil((T)nf1 / bin_size_x);

  T *d_kx               = d_plan->kx;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  T sigma = d_plan->opts.upsampfac;

  const auto sharedplanorysize = shared_memory_required<T>(
      1, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);

  if (d_plan->opts.gpu_kerevalmeth) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(interp_1d_output_driven<T, 1, ns>, 1, *d_plan) !=
            0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      interp_1d_output_driven<T, 1, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_binstartpts,
          d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts, d_plan->opts.gpu_np);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(interp_1d_output_driven<T, 0, ns>, 1, *d_plan) !=
            0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      interp_1d_output_driven<T, 0, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_binstartpts,
          d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts, d_plan->opts.gpu_np);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template int cuinterp1d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp1d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft
