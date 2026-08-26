// Tests for the modern C++ GPU interface (include/cufinufft.hpp, C++20).
// Answer checks copy device buffers back to the host and compare against
// direct O(NM) sums; misuse cases must throw cufinufft::error with the
// expected FINUFFT_ERR_* code. This translation unit also includes
// finufft.hpp to prove the CPU and GPU headers co-exist (type-level checks
// only; linking the CPU library is not needed for those).
// Exit code 0 iff every check passes.

#include <cufinufft.hpp>
#include <finufft.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <string>
#include <type_traits>
#include <vector>

namespace {

int fails = 0;

void report(bool ok, const std::string &name) {
  if (!ok) {
    ++fails;
    std::cout << "  FAIL " << name << '\n';
  }
}

// The CPU header co-includes: same class shape, disjoint namespaces.
static_assert(
    std::is_same_v<decltype(finufft::plan(1, {8}, +1, 1e-6)), finufft::plan<double>>);
static_assert(
    std::is_same_v<decltype(cufinufft::plan(1, {8}, +1, 1e-6f)), cufinufft::plan<float>>);

// Deterministic generator on [0,1).
template<typename T> T unif(unsigned &st) {
  st ^= st << 13;
  st ^= st >> 17;
  st ^= st << 5;
  return T(st & 0xFFFFFFu) / T(0x1000000u);
}

template<typename T> void fill_pts(std::vector<T> &p, unsigned &st) {
  for (auto &v : p) v = std::numbers::pi_v<T> * (T(2) * unif<T>(st) - T(1));
}

template<typename T> void fill_cx(std::vector<std::complex<T>> &c, unsigned &st) {
  for (auto &v : c) v = {T(2) * unif<T>(st) - T(1), T(2) * unif<T>(st) - T(1)};
}

// Device buffer helper: RAII around cudaMalloc/cudaFree.
template<typename T> struct devbuf {
  T *p = nullptr;
  explicit devbuf(std::size_t n) { cudaMalloc(&p, n * sizeof(T)); }
  ~devbuf() { cudaFree(p); }
  devbuf(const devbuf &)            = delete;
  devbuf &operator=(const devbuf &) = delete;
};
template<typename T> void to_dev(devbuf<T> &d, const std::vector<T> &h, std::size_t n) {
  cudaMemcpy(d.p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
}
template<typename T> void to_host(std::vector<T> &h, const devbuf<T> &d, std::size_t n) {
  cudaMemcpy(h.data(), d.p, n * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
double relerr(const std::vector<std::complex<T>> &got,
              const std::vector<std::complex<T>> &ref) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    num = std::max(num, double(std::abs(got[i] - ref[i])));
    den = std::max(den, double(std::abs(ref[i])));
  }
  return num / den;
}

// 1d type 1, double, checked against the direct sum for every mode.
void run_1d1() {
  const std::int64_t M = 97, N = 16;
  unsigned st = 7u;
  std::vector<double> x(M);
  fill_pts(x, st);
  std::vector<std::complex<double>> c(M);
  fill_cx(c, st);
  devbuf<double> dx(M);
  devbuf<std::complex<double>> dc(M), dF(N);
  to_dev(dx, x, M);
  to_dev(dc, c, M);

  cufinufft::plan p(1, {N}, +1, 1e-9);
  p.setpts(std::span<const double>(dx.p, M));
  p.execute(std::span<std::complex<double>>(dc.p, M),
            std::span<std::complex<double>>(dF.p, N));

  std::vector<std::complex<double>> F(N);
  to_host(F, dF, N);
  const std::complex<double> I(0, 1);
  double num = 0, den = 0;
  for (std::int64_t i = 0; i < N; ++i) {
    const double k = double(i) - N / 2.0;
    std::complex<double> acc(0, 0);
    for (std::int64_t j = 0; j < M; ++j) acc += c[j] * std::exp(I * k * x[j]);
    num = std::max(num, std::abs(F[i] - acc));
    den = std::max(den, std::abs(acc));
  }
  report(num / den < 1e-7, "gpu 1d1 accuracy");
}

// 1d type 3, float: points and target freqs both go through variadic setpts.
void run_1d3() {
  const std::int64_t M = 63, nk = 41;
  unsigned st = 21u;
  std::vector<float> x(M), s(nk);
  fill_pts(x, st);
  fill_pts(s, st);
  std::vector<std::complex<float>> c(M);
  fill_cx(c, st);
  devbuf<float> dx(M), ds(nk);
  devbuf<std::complex<float>> dc(M), dF(nk);
  to_dev(dx, x, M);
  to_dev(ds, s, nk);
  to_dev(dc, c, M);

  cufinufft::plan p(3, {0}, +1, 1e-4f);
  p.setpts(std::span<const float>(dx.p, M), std::span<const float>(ds.p, nk));
  p.execute(std::span<std::complex<float>>(dc.p, M),
            std::span<std::complex<float>>(dF.p, nk));

  std::vector<std::complex<float>> F(nk);
  to_host(F, dF, nk);
  const std::complex<float> I(0, 1);
  double num = 0, den = 0;
  for (std::int64_t k = 0; k < nk; ++k) {
    std::complex<float> acc(0, 0);
    for (std::int64_t j = 0; j < M; ++j) acc += c[j] * std::exp(I * s[k] * x[j]);
    num = std::max(num, double(std::abs(F[k] - acc)));
    den = std::max(den, double(std::abs(acc)));
  }
  report(num / den < 1e-2, "gpu 1d3 accuracy");
}

// 2d type 1, float: two source spans select two dimensions.
void run_2d1() {
  const std::int64_t M = 97, N1 = 8, N2 = 10;
  unsigned st = 33u;
  std::vector<float> x(M), y(M);
  fill_pts(x, st);
  fill_pts(y, st);
  std::vector<std::complex<float>> c(M);
  fill_cx(c, st);
  devbuf<float> dx(M), dy(M);
  devbuf<std::complex<float>> dc(M), dF(N1 * N2);
  to_dev(dx, x, M);
  to_dev(dy, y, M);
  to_dev(dc, c, M);

  cufinufft::plan p(1, {N1, N2}, +1, 1e-3f);
  p.setpts(std::span<const float>(dx.p, M), std::span<const float>(dy.p, M));
  p.execute(std::span<std::complex<float>>(dc.p, M),
            std::span<std::complex<float>>(dF.p, N1 * N2));

  std::vector<std::complex<float>> F(N1 * N2);
  to_host(F, dF, N1 * N2);
  const std::complex<float> If(0, 1);
  double num = 0, den = 0;
  for (std::int64_t i1 = 0; i1 < N1; ++i1) {
    for (std::int64_t i2 = 0; i2 < N2; ++i2) {
      const float k1 = float(i1) - N1 / 2.f, k2 = float(i2) - N2 / 2.f;
      std::complex<float> acc(0, 0);
      for (std::int64_t j = 0; j < M; ++j)
        acc += c[j] * std::exp(If * (k1 * x[j] + k2 * y[j]));
      const auto idx = std::size_t(i1 + N1 * i2);
      num            = std::max(num, double(std::abs(F[idx] - acc)));
      den            = std::max(den, double(std::abs(acc)));
    }
  }
  report(num / den < 1e-2, "gpu 2d1 accuracy");
}

void expect_error(int want_code, const char *name, void (*fn)()) {
  try {
    fn();
    report(false, name);
  } catch (const cufinufft::error &e) {
    report(e.code() == want_code,
           std::string(name) + " (code " + std::to_string(e.code()) + ")");
  } catch (const std::exception &e) {
    report(false, std::string(name) + " (wrong type: " + e.what() + ")");
  }
}

void bad_type() { cufinufft::plan<double> p(4, {16}, +1, 1e-6); }
void not_ready() {
  devbuf<std::complex<double>> bc(31), bout(16);
  cufinufft::plan<double> p(1, {16}, +1, 1e-6);
  p.execute(std::span<std::complex<double>>(bc.p, 31),
            std::span<std::complex<double>>(bout.p, 16));
}
void moved_from() {
  devbuf<double> bx(31);
  cufinufft::plan<double> p(1, {16}, +1, 1e-6);
  cufinufft::plan<double> q = std::move(p);
  p.setpts(std::span<const double>(bx.p, 31)); // p is moved-from: null handle
}
void wrong_arity() {
  devbuf<double> bx(31);
  cufinufft::plan<double> p(1, {16, 16}, +1, 1e-6);
  p.setpts(std::span<const double>(bx.p, 31)); // a 2d plan needs two spans
}
void wrong_io_size() {
  devbuf<double> bx(31);
  devbuf<std::complex<double>> bc(31), boutshort(15);
  cufinufft::plan<double> p(1, {16}, +1, 1e-6);
  p.setpts(std::span<const double>(bx.p, 31));
  p.execute(std::span<std::complex<double>>(bc.p, 31),
            std::span<std::complex<double>>(boutshort.p, 15));
}

} // namespace

int main() {
  if (int ndev = 0; cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    std::cout << "cufinufft_cpp_interface: no CUDA device, skipping\n";
    return 77; // ctest SKIP_RETURN_CODE
  }
  run_1d1();
  run_1d3();
  run_2d1();
  expect_error(FINUFFT_ERR_TYPE_NOTVALID, "bad type", &bad_type);
  expect_error(FINUFFT_ERR_INVALID_ARGUMENT, "execute before setpts", &not_ready);
  expect_error(FINUFFT_ERR_PLAN_NOTVALID, "moved-from plan", &moved_from);
  expect_error(FINUFFT_ERR_INVALID_ARGUMENT, "wrong setpts arity", &wrong_arity);
  expect_error(FINUFFT_ERR_INVALID_ARGUMENT, "execute buffer size", &wrong_io_size);
  if (fails) {
    std::cout << "cufinufft_cpp_interface: " << fails << " check(s) failed\n";
    return 1;
  }
  std::cout << "cufinufft_cpp_interface: all checks passed\n";
  return 0;
}
