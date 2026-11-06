// Dumb-input cases for the cuFINUFFT public C API (issue #947): empty point
// sets (M=0 or nk=0) and empty mode sets (any N_i=0) are valid and must
// return 0, with the output exactly zero where it has elements. The output
// is c for type 2 and f otherwise. Only Thrust allocates, poisons and reads
// back device memory; no raw CUDA memory calls.
//
// Prints a line per case and exits non-zero on the first failure.

#include <cstdint>
#include <cstdio>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <cufinufft.h>

namespace {

constexpr double TOL = 1e-9;

// One dumb-input case: makeplan, setpts and execute must all return 0 and the
// output must be exactly zero where it has elements. zero_dim < 0 keeps all
// modes at 4, otherwise that mode count is 0. spreadinterponly sets
// opts.gpu_spreadinterponly for the case.
int empty_case(int type, int dim, int64_t M, int64_t nk, int zero_dim,
               int spreadinterponly = 0) {
  int64_t nmodes[3] = {4, 4, 4};
  if (zero_dim >= 0) nmodes[zero_dim] = 0;
  std::int64_t nfk = 1;
  for (int d = 0; d < dim; ++d) nfk *= nmodes[d];
  // Elements the plan must write: f for types 1 and 3, c for type 2.
  const std::int64_t n_out = (type == 2) ? M : (type == 3 ? nk : nfk);

  // Points/targets, empty where the case has none.
  thrust::device_vector<double> d_x(M), d_y(dim >= 2 ? M : 0), d_z(dim >= 3 ? M : 0);
  thrust::device_vector<double> d_s(type == 3 ? nk : 0),
      d_t(type == 3 && dim >= 2 ? nk : 0), d_u(type == 3 && dim >= 3 ? nk : 0);
  thrust::fill(d_x.begin(), d_x.end(), 0.5);
  thrust::fill(d_y.begin(), d_y.end(), 0.5);
  thrust::fill(d_z.begin(), d_z.end(), 0.5);
  thrust::fill(d_s.begin(), d_s.end(), 0.5);
  thrust::fill(d_t.begin(), d_t.end(), 0.5);
  thrust::fill(d_u.begin(), d_u.end(), 0.5);

  // c: input for types 1 and 3, zeroed output for type 2.
  thrust::device_vector<cuDoubleComplex> d_c(M);
  // f: zeroed output for types 1 and 3, arbitrary input for type 2.
  thrust::device_vector<cuDoubleComplex> d_fk(type == 3 ? nk : nfk);
  thrust::fill(d_c.begin(), d_c.end(), cuDoubleComplex{1.0, 0.0});
  thrust::fill(d_fk.begin(), d_fk.end(), cuDoubleComplex{1.0, 0.0});
  // Poison the output so a wrong success cannot pass the zero check.
  const bool out_is_c = (type == 2);
  auto &out           = out_is_c ? d_c : d_fk;
  if (n_out > 0) thrust::fill(out.begin(), out.end(), cuDoubleComplex{0.25, -0.25});

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_spreadinterponly = spreadinterponly;
  cufinufft_plan plan{};
  int ier = cufinufft_makeplan(type, dim, nmodes, 1, 1, TOL, &plan, &opts);
  if (!ier)
    ier = cufinufft_setpts(
        plan, M, d_x.data().get(), dim >= 2 ? d_y.data().get() : nullptr,
        dim >= 3 ? d_z.data().get() : nullptr, type == 3 ? nk : 0,
        type == 3 ? d_s.data().get() : nullptr, type == 3 ? d_t.data().get() : nullptr,
        type == 3 ? d_u.data().get() : nullptr);
  if (!ier) ier = cufinufft_execute(plan, d_c.data().get(), d_fk.data().get());
  const int ier_destroy = cufinufft_destroy(plan);
  if (!ier) ier = ier_destroy;
  // Catch a sticky launch error: with no output, no later CUDA call would.
  const int ier_cuda = cudaGetLastError() != cudaSuccess;
  int nnz            = 0;
  if (!ier && n_out > 0) {
    thrust::host_vector<cuDoubleComplex> h_out(out.size());
    thrust::copy(out.begin(), out.end(), h_out.begin());
    for (const auto &z : h_out) nnz += (z.x != 0.0 || z.y != 0.0);
  }
  if (ier || ier_cuda || nnz) {
    std::fprintf(stderr,
                 "dumb input case failed: type=%d dim=%d M=%ld nk=%ld zdim=%d so=%d "
                 "ier=%d cuda_err=%d nnz=%d\n",
                 type, dim, (long)M, (long)nk, zero_dim, spreadinterponly, ier, ier_cuda,
                 nnz);
    return 1;
  }
  return 0;
}

} // namespace

int main() {
  int rc = 0;
  // type 1: M=0 zeros f; N1=0 is a no-op returning 0.
  rc |= empty_case(1, 1, 0, 0, -1);
  rc |= empty_case(1, 1, 3, 0, 0);
  // type 2: M=0 is a no-op returning 0; N1=0 zeros c.
  rc |= empty_case(2, 1, 0, 0, -1);
  rc |= empty_case(2, 1, 3, 0, 0);
  // type 3: nj=0 zeros f; nk=0 is a no-op returning 0.
  rc |= empty_case(3, 1, 0, 3, -1);
  rc |= empty_case(3, 1, 3, 0, -1);
  // type 2, 3D: a zero in a non-first dimension zeros c.
  rc |= empty_case(2, 3, 3, 0, 2);
  // spreadinterponly path: makeplan and execute return 0; type 2 zeros c.
  rc |= empty_case(1, 1, 3, 0, 0, 1);
  rc |= empty_case(2, 1, 3, 0, 0, 1);
  if (!rc) std::printf("cufinufft_dumbinputs: ok (9 cases)\n");
  return rc;
}
