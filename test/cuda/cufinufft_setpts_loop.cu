#include <cuda_runtime.h>
#include <curand.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cufinufft.h>

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

__global__ void scale_shift(double *data, int64_t n, double scale, double shift) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = data[idx] * scale + shift;
}

void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
    std::exit(1);
  }
}

void check_curand(curandStatus_t err, const char *msg) {
  if (err != CURAND_STATUS_SUCCESS) {
    std::cerr << msg << ": curand error " << static_cast<int>(err) << "\n";
    std::exit(1);
  }
}

void check_cufinufft(int err, const char *msg) {
  if (err != 0) {
    std::cerr << msg << ": cufinufft error " << err << "\n";
    std::exit(1);
  }
}
} // namespace

int main(int argc, char **argv) {
  int64_t M          = 100000;
  int64_t iterations = 1000000000LL;
  int64_t log_every  = 100;

  if (argc > 1) M = std::atoll(argv[1]);
  if (argc > 2) iterations = std::atoll(argv[2]);
  if (argc > 3) log_every = std::atoll(argv[3]);

  std::cout << "Starting, please run `nvidia-smi -lms 500` to watch VRAM utilization.\n";
  std::cout << "M=" << M << " iterations=" << iterations << " log_every=" << log_every
            << "\n";

  // Allocate NU point arrays on device.
  double *d_x = nullptr;
  double *d_y = nullptr;
  check_cuda(cudaMalloc(&d_x, sizeof(double) * M), "cudaMalloc d_x failed");
  check_cuda(cudaMalloc(&d_y, sizeof(double) * M), "cudaMalloc d_y failed");

  // Setup CURAND generator.
  curandGenerator_t gen;
  check_curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT),
               "curandCreateGenerator failed");
  check_curand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL),
               "curandSetPseudoRandomGeneratorSeed failed");

  // Create a 2D type-1 plan with 256x256 modes.
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  const int type = 1;
  const int dim  = 2;
  const int iflag = 1;
  const int ntr = 1;
  const double eps = 1e-6;
  int64_t nmodes[2] = {256, 256};
  cufinufft_plan plan = nullptr;

  check_cufinufft(cufinufft_makeplan(type, dim, nmodes, iflag, ntr, eps, &plan, &opts),
                  "cufinufft_makeplan failed");

  int64_t iter = 0;
  size_t free_bytes = 0, total_bytes = 0;

  while (iter < iterations) {
    // Generate random points in [-pi, pi].
    check_curand(curandGenerateUniformDouble(gen, d_x, M),
                 "curandGenerateUniformDouble d_x failed");
    check_curand(curandGenerateUniformDouble(gen, d_y, M),
                 "curandGenerateUniformDouble d_y failed");

    const int threads = 256;
    int blocks = static_cast<int>((M + threads - 1) / threads);
    scale_shift<<<blocks, threads>>>(d_x, M, 2.0 * kPi, -kPi);
    scale_shift<<<blocks, threads>>>(d_y, M, 2.0 * kPi, -kPi);
    check_cuda(cudaGetLastError(), "scale_shift kernel launch failed");

    // Call setpts repeatedly on the same plan.
    check_cufinufft(cufinufft_setpts(plan, M, d_x, d_y, nullptr, 0, nullptr, nullptr, nullptr),
                    "cufinufft_setpts failed");

    if (log_every > 0 && (iter % log_every == 0)) {
      check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo failed");
      double free_gb  = static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0);
      double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
      std::cout << "iter=" << iter << " free_gb=" << free_gb << " total_gb=" << total_gb
                << "\n";
    }

    ++iter;
  }

  check_cufinufft(cufinufft_destroy(plan), "cufinufft_destroy failed");
  check_curand(curandDestroyGenerator(gen), "curandDestroyGenerator failed");
  check_cuda(cudaFree(d_x), "cudaFree d_x failed");
  check_cuda(cudaFree(d_y), "cudaFree d_y failed");

  return 0;
}
