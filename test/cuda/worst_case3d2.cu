#include <cmath>
#include <complex.h>
#include <complex>
#include <cufinufft/contrib/helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <random>

#include <cufinufft.h>
#include <finufft.h>

#include <cufinufft/impl.h>
#include <cufinufft/utils.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../utils/dirft3d.hpp"
#include "../utils/norms.hpp"

#include <fstream>

constexpr auto TEST_BIGPROB = 1e8;

template<typename T>
void read_data(const std::string &filename, thrust::host_vector<T> &data, size_t size) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  data.resize(size);
  file.read(reinterpret_cast<char *>(data.data()), size * sizeof(T));
  file.close();
}

void read_parameters(const std::string &filename, int &nf1, int &nf2, int &nf3, int &M) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.find("nf1:") != std::string::npos)
      nf1 = std::stoi(line.substr(line.find(":") + 1));
    if (line.find("nf2:") != std::string::npos)
      nf2 = std::stoi(line.substr(line.find(":") + 1));
    if (line.find("nf3:") != std::string::npos)
      nf3 = std::stoi(line.substr(line.find(":") + 1));
    if (line.find("M:") != std::string::npos)
      M = std::stoi(line.substr(line.find(":") + 1));
  }
  file.close();
}

template<typename T>
int run_test(int method, int type, int N1, int N2, int N3, int M, T tol, T checktol,
             int iflag, double upsampfac) {
  std::cout << std::scientific << std::setprecision(14);
  read_parameters("params.txt", N1, N2, N3, M);

  thrust::host_vector<T> x(M), y(M), z(M), s{}, t{}, u{};
  thrust::host_vector<thrust::complex<T>> c(M), fk(N1 * N2 * N3);

  thrust::device_vector<T> d_x(M), d_y(M), d_z(M), d_s{}, d_t{}, d_u{};
  thrust::device_vector<thrust::complex<T>> d_c(M), d_fk(N1 * N2 * N3);

  std::default_random_engine eng(1);
  std::uniform_real_distribution<T> dist11(0, 1);
  auto randm11 = [&eng, &dist11]() {
    return dist11(eng);
  };

  read_data("kx.bin", x, M);
  read_data("ky.bin", y, M);
  read_data("kz.bin", z, M);
  // read_data("c.bin", c, M);
  read_data("fk.bin", fk, N1 * N2 * N3);

  d_x  = x;
  d_y  = y;
  d_z  = z;
  d_fk = fk;

  cufinufft_plan_t<T> *dplan;
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_method      = 1;
  opts.gpu_kerevalmeth = 1;
  opts.upsampfac       = 1.25;
  opts.debug           = 2;
  int dim              = 3;
  int nmodes[3]        = {N1, N2, N3};
  int ntransf          = 1;

  finufft_opts opts_finufft;
  finufft_default_opts(&opts_finufft);
  opts_finufft.upsampfac          = opts.upsampfac;
  opts_finufft.spread_kerevalmeth = opts.gpu_kerevalmeth;
  cufinufft_makeplan_impl(type, dim, nmodes, iflag, ntransf, tol, &dplan, &opts);
  cufinufft_setpts_impl<T>(M, d_x.data().get(), d_y.data().get(), d_z.data().get(),
                           N1 * N2 * N3, nullptr, nullptr, nullptr, dplan);
  cufinufft_execute_impl<T>((cuda_complex<T> *)d_c.data().get(),
                            (cuda_complex<T> *)d_fk.data().get(), dplan);
  cudaDeviceSynchronize();
  thrust::host_vector<thrust::complex<T>> gpu_c = d_c;
  cudaDeviceSynchronize();
  cufinufft_destroy_impl<T>(dplan);

  finufft3d2(M, x.data(), y.data(), z.data(), (std::complex<T> *)c.data(), iflag, tol, N1,
             N2, N3, (std::complex<T> *)fk.data(), nullptr);
  std::cout << "CPU ----------------------------" << std::endl;
  for (auto jt = 0; jt < M; ++jt) {
    T rel_error           = std::numeric_limits<T>::min();
    thrust::complex<T> J  = thrust::complex<T>(0, iflag);
    thrust::complex<T> ct = thrust::complex<T>(0, 0);
    int m                 = 0;
    for (int m3 = -(N3 / 2); m3 <= (N3 - 1) / 2; ++m3)   // loop in correct order over F
      for (int m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order
                                                         // over F
        for (int m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
          ct += fk[m++] * exp(J * (m1 * x[jt] + m2 * y[jt] + m3 * z[jt])); // crude direct
    rel_error = thrust::abs(ct - c[jt]) / thrust::abs(ct);
    if (rel_error > tol * 10) {
      std::cout << "[ref   ] ct = " << ct << std::endl;
      std::cout << "[cpu   ] c[" << jt << "] = " << c[jt] << std::endl;
      printf("[cpu   ] one targ: rel err in c[%ld] is %.3g\n", (int64_t)jt, rel_error);
    }
  }
  std::cout << "GPU ----------------------------" << std::endl;
  for (auto jt = 0; jt < M; ++jt) {
    T rel_error           = std::numeric_limits<T>::min();
    thrust::complex<T> J  = thrust::complex<T>(0, iflag);
    thrust::complex<T> ct = thrust::complex<T>(0, 0);
    int m                 = 0;
    for (int m3 = -(N3 / 2); m3 <= (N3 - 1) / 2; ++m3)   // loop in correct order over F
      for (int m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order
                                                         // over F
        for (int m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
          ct += fk[m++] * exp(J * (m1 * x[jt] + m2 * y[jt] + m3 * z[jt])); // crude direct
    rel_error = thrust::abs(ct - gpu_c[jt]) / thrust::abs(ct);
    if (rel_error > tol * 10) {
      std::cout << "[gpu   ] gpu[" << jt << "] = " << gpu_c[jt] << std::endl;
      printf("[gpu   ] one targ: rel err in c[%ld] is %.3g\n", (int64_t)jt, rel_error);
    }
  }

  for (auto jt = 0; jt < M; ++jt) {
    auto error = thrust::abs(c[jt] - gpu_c[jt]);
    std::cout << "[abs(diff) [" << jt << " ] " << error << std::endl;
  }

  // if (rel_error > checktol) {
  // printf("[gpu   ]\terr %.3e > checktol %.3e\n", rel_error, checktol);
  // }

  return 0;
}

int main(int argc, char *argv[]) {
  // if (argc != 11) {
  //   fprintf(stderr,
  //           "Usage: cufinufft3d1_test method type N1 N2 N3 M tol checktol prec\n"
  //           "Arguments:\n"
  //           "  method: One of\n"
  //           "    1: nupts driven,\n"
  //           "    2: sub-problem, or\n"
  //           "    4: block gather.\n"
  //           "  type: Type of transform (1, 2, 3)"
  //           "  N1, N2, N3: The size of the 3D array\n"
  //           "  M: The number of non-uniform points\n"
  //           "  tol: NUFFT tolerance\n"
  //           "  checktol:  relative error to pass test\n"
  //           "  prec:  'f' or 'd' (float/double)\n"
  //           "  upsamplefac: upsampling factor\n");
  //   return 1;
  // }
  // const int method       = atoi(argv[1]);
  // const int type         = atoi(argv[2]);
  // const int N1           = atof(argv[3]);
  // const int N2           = atof(argv[4]);
  // const int N3           = atof(argv[5]);
  // const int M            = atof(argv[6]);
  // const double tol       = atof(argv[7]);
  // const double checktol  = atof(argv[8]);
  // const char prec        = argv[9][0];
  // const double upsampfac = atof(argv[10]);
  // const int iflag        = 1;
  return run_test<double>(1, 2, 2, 5, 10, 16, 1e-8, 1e-7, 1, 1.25);
  return -1;
}
