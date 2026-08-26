// Tests for the modern C++ CPU interface (include/finufft.hpp, C++20).
// Answers are checked against direct O(NM) sums; misuse cases (the "bend"
// section) must throw finufft::error with the expected FINUFFT_ERR_* code.
// Exit code 0 iff every check passes.

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

// Deterministic generator on [0,1), independent of the C library rand().
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

// Mode index k for output array index i, CMCL ordering (modeord=0):
// k = i - ms/2, covering -ms/2 .. ms-1-ms/2 (ms is even in these tests).
template<int D>
std::array<double, D> mode_at(std::int64_t lin, const std::array<int, D> &ms) {
  std::array<double, D> k{};
  for (int d = 0; d < D; ++d) {
    k[d] = double(lin % ms[d]) - ms[d] / 2.0;
    lin /= ms[d];
  }
  return k;
}

// Direct references, iflag sign convention matching the library.
template<typename T, int D>
std::vector<std::complex<T>> direct1(const std::array<std::vector<T>, D> &x,
                                     const std::vector<std::complex<T>> &c,
                                     const std::array<int, D> &ms, int iflag) {
  const std::complex<T> I(T(0), T(1));
  std::int64_t N = 1;
  for (int d = 0; d < D; ++d) N *= ms[d];
  std::vector<std::complex<T>> fk(static_cast<std::size_t>(N));
  const auto M = std::int64_t(c.size());
  for (std::int64_t i = 0; i < N; ++i) {
    const auto k = mode_at<D>(i, ms);
    std::complex<T> acc(0, 0);
    for (std::int64_t j = 0; j < M; ++j) {
      T ph = 0;
      for (int d = 0; d < D; ++d) ph += T(k[d]) * x[d][std::size_t(j)];
      acc += c[std::size_t(j)] * std::exp(I * T(iflag) * ph);
    }
    fk[std::size_t(i)] = acc;
  }
  return fk;
}

template<typename T, int D>
std::vector<std::complex<T>> direct2(
    const std::array<std::vector<T>, D> &x, const std::vector<std::complex<T>> &fk,
    const std::array<int, D> &ms, int iflag, std::int64_t M) {
  const std::complex<T> I(T(0), T(1));
  std::vector<std::complex<T>> c(static_cast<std::size_t>(M));
  std::int64_t N = 1;
  for (int d = 0; d < D; ++d) N *= ms[d];
  for (std::int64_t j = 0; j < M; ++j) {
    std::complex<T> acc(0, 0);
    for (std::int64_t i = 0; i < N; ++i) {
      const auto k = mode_at<D>(i, ms);
      T ph         = 0;
      for (int d = 0; d < D; ++d) ph += T(k[d]) * x[d][std::size_t(j)];
      acc += fk[std::size_t(i)] * std::exp(I * T(iflag) * ph);
    }
    c[std::size_t(j)] = acc;
  }
  return c;
}

template<typename T, int D>
std::vector<std::complex<T>> direct3(const std::array<std::vector<T>, D> &x,
                                     const std::vector<std::complex<T>> &c,
                                     const std::array<std::vector<T>, D> &s, int iflag) {
  const std::complex<T> I(T(0), T(1));
  const auto M  = std::int64_t(c.size());
  const auto nk = std::int64_t(s[0].size());
  std::vector<std::complex<T>> fk(static_cast<std::size_t>(nk));
  for (std::int64_t k = 0; k < nk; ++k) {
    std::complex<T> acc(0, 0);
    for (std::int64_t j = 0; j < M; ++j) {
      T ph = 0;
      for (int d = 0; d < D; ++d) ph += s[d][std::size_t(k)] * x[d][std::size_t(j)];
      acc += c[std::size_t(j)] * std::exp(I * T(iflag) * ph);
    }
    fk[std::size_t(k)] = acc;
  }
  return fk;
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

// One transform, type 1/2/3 in D dims, checked against the direct sums.
template<typename T, int D> void run_type(int type, const char *tag) {
  const auto ms = [] {
    if constexpr (D == 1)
      return std::array<int, D>{10};
    else if constexpr (D == 2)
      return std::array<int, D>{10, 12};
    else
      return std::array<int, D>{10, 12, 14};
  }();
  std::int64_t N = 1;
  for (int d = 0; d < D; ++d) N *= ms[d];
  const std::int64_t M = 97, nk = 61;
  const T tol = std::is_same_v<T, float> ? T(1e-4) : T(1e-12);
  const double bound =
      double(std::is_same_v<T, float> ? T(1e3) : T(1e2)) * double(tol); // sanity margin
  unsigned st = 12345u + unsigned(100 * D + 10 * type);

  std::array<std::vector<T>, D> x, s;
  for (int d = 0; d < D; ++d) {
    x[d].resize(static_cast<std::size_t>(M));
    fill_pts(x[d], st);
    s[d].resize(static_cast<std::size_t>(nk));
    fill_pts(s[d], st);
  }
  std::vector<std::complex<T>> c(static_cast<std::size_t>(M));
  fill_cx(c, st);
  std::vector<std::complex<T>> ref;
  if (type == 3) ref = direct3<T, D>(x, c, s, +1);
  if (type == 1) ref = direct1<T, D>(x, c, ms, +1);

  std::array<std::int64_t, D> m64;
  for (int d = 0; d < D; ++d) m64[d] = ms[d];
  finufft::plan<T> p(type, std::span<const std::int64_t>(m64.data(), D), +1, tol);

  if (type == 3) {
    if constexpr (D == 1)
      p.setpts(x[0], s[0]);
    else if constexpr (D == 2)
      p.setpts(x[0], x[1], s[0], s[1]);
    else
      p.setpts(x[0], x[1], x[2], s[0], s[1], s[2]);
  } else {
    if constexpr (D == 1)
      p.setpts(x[0]);
    else if constexpr (D == 2)
      p.setpts(x[0], x[1]);
    else
      p.setpts(x[0], x[1], x[2]);
  }

  // For the adjoint the data flow reverses: a type-2 adjoint maps NU values
  // to modes, a type-1 adjoint maps modes to NU values, both with the
  // exponential sign conjugated.
  if (type == 2) {
    std::vector<std::complex<T>> fkin(static_cast<std::size_t>(N));
    fill_cx(fkin, st);
    std::vector<std::complex<T>> cout(static_cast<std::size_t>(M));
    ref = direct2<T, D>(x, fkin, ms, +1, M);
    p.execute(cout, fkin);
    report(relerr(cout, ref) < bound, std::string(tag) + " type2 accuracy");
    std::vector<std::complex<T>> fadj(static_cast<std::size_t>(N));
    p.execute_adjoint(c, fadj);
    const auto refadj = direct1<T, D>(x, c, ms, -1);
    report(relerr(fadj, refadj) < bound, std::string(tag) + " type2 adjoint accuracy");
    return;
  }
  std::vector<std::complex<T>> out(static_cast<std::size_t>(type == 3 ? nk : N));
  p.execute(c, out);
  report(relerr(out, ref) < bound,
         std::string(tag) + " type" + std::to_string(type) + " accuracy");
  if (type == 1) {
    std::vector<std::complex<T>> fkin(static_cast<std::size_t>(N));
    fill_cx(fkin, st);
    std::vector<std::complex<T>> cadj(static_cast<std::size_t>(M));
    p.execute_adjoint(cadj, fkin);
    const auto refadj = direct2<T, D>(x, fkin, ms, -1, M);
    report(relerr(cadj, refadj) < bound, std::string(tag) + " type1 adjoint accuracy");
  }
}

// ntrans=2 (many) accuracy check: two packed transforms in one execute call.
template<typename T> void run_many(const char *tag) {
  const std::int64_t M = 63, N = 16;
  const T tol        = std::is_same_v<T, float> ? T(1e-4) : T(1e-12);
  const double bound = 1e2 * double(tol);
  unsigned st        = 999u;
  std::vector<T> x(static_cast<std::size_t>(M));
  fill_pts(x, st);
  std::vector<std::complex<T>> c0(static_cast<std::size_t>(M)),
      c1(static_cast<std::size_t>(M));
  fill_cx(c0, st);
  fill_cx(c1, st);
  std::vector<std::complex<T>> cpk = c0;
  cpk.insert(cpk.end(), c1.begin(), c1.end()); // transform-major packing
  std::vector<std::complex<T>> out(std::size_t(2 * N));
  finufft::plan<T> p(1, {N}, +1, tol, 2);
  p.setpts(x);
  p.execute(cpk, out);
  const std::array<std::vector<T>, 1> xa{x};
  const auto r0 = direct1<T, 1>(xa, c0, {int(N)}, +1);
  const auto r1 = direct1<T, 1>(xa, c1, {int(N)}, +1);
  std::vector<std::complex<T>> o0(out.begin(), out.begin() + N),
      o1(out.begin() + N, out.end());
  report(relerr(o0, r0) < bound && relerr(o1, r1) < bound,
         std::string(tag) + " ntrans=2 accuracy");
}

// Compile-time checks: deduction guides pick the precision off the tolerance.
void run_ctad() {
  finufft::plan cd(1, {8}, +1, 1e-6);
  static_assert(std::is_same_v<decltype(cd), finufft::plan<double>>);
  finufft::plan cf(1, {8}, +1, 1e-6f);
  static_assert(std::is_same_v<decltype(cf), finufft::plan<float>>);
  const std::array<std::int64_t, 2> m{8, 8};
  finufft::plan cs(2, m, -1, 1e-3f);
  static_assert(std::is_same_v<decltype(cs), finufft::plan<float>>);
  report(true, "ctad");
}

void expect_error(int want_code, const char *name, void (*fn)()) {
  try {
    fn();
    report(false, name);
  } catch (const finufft::error &e) {
    report(e.code() == want_code,
           std::string(name) + " (code " + std::to_string(e.code()) + ")");
  } catch (const std::exception &e) {
    report(false, std::string(name) + " (wrong type: " + e.what() + ")");
  }
}

// Each bend case is a small function so expect_error can call it uniformly.
std::vector<double> bx(31), by(31), byshort(30);
std::vector<std::complex<double>> bc(31), bout(16), boutshort(15);

void bad_type() { finufft::plan<double> p(4, {16}, +1, 1e-6); }
void bad_dim() { finufft::plan<double> p(1, {}, +1, 1e-6); }
void eps_small() {
  finufft::plan<double> p(1, {16}, +1, 1e-30);
  p.setpts(bx);
}
void not_ready() {
  finufft::plan<double> p(1, {16}, +1, 1e-6);
  p.execute(bc, bout);
}
void moved_from() {
  finufft::plan<double> p(1, {16}, +1, 1e-6);
  finufft::plan<double> q = std::move(p);
  p.setpts(bx); // p is moved-from: its handle is null
}
void move_target_works() {
  finufft::plan<double> p(1, {16}, +1, 1e-6);
  finufft::plan<double> q = std::move(p);
  q.setpts(bx);
  q.execute(bc, bout);
}
void wrong_arity() {
  finufft::plan<double> p(1, {16, 16}, +1, 1e-6);
  p.setpts(bx); // a 2d plan needs two span arguments
}
void size_mismatch() {
  finufft::plan<double> p(1, {16, 16}, +1, 1e-6);
  p.setpts(bx, byshort);
}
void wrong_io_size() {
  finufft::plan<double> p(1, {16}, +1, 1e-6);
  p.setpts(bx);
  p.execute(bc, boutshort);
}

void run_bend() {
  expect_error(FINUFFT_ERR_TYPE_NOTVALID, "bad type", &bad_type);
  expect_error(FINUFFT_ERR_DIM_NOTVALID, "bad dim", &bad_dim);
  expect_error(FINUFFT_ERR_EPS_TOO_SMALL, "eps too small", &eps_small);
  expect_error(FINUFFT_ERR_INVALID_ARGUMENT, "execute before setpts", &not_ready);
  expect_error(FINUFFT_ERR_PLAN_NOTVALID, "moved-from plan", &moved_from);
  move_target_works(); // must not throw
  expect_error(FINUFFT_ERR_INVALID_ARGUMENT, "wrong setpts arity", &wrong_arity);
  expect_error(FINUFFT_ERR_INVALID_ARGUMENT, "setpts size mismatch", &size_mismatch);
  expect_error(FINUFFT_ERR_INVALID_ARGUMENT, "execute buffer size", &wrong_io_size);
}

} // namespace

int main() {
  run_ctad();
  run_many<double>("double");
  run_many<float>("float");
  run_type<double, 1>(1, "double 1d");
  run_type<double, 1>(2, "double 1d");
  run_type<double, 1>(3, "double 1d");
  run_type<double, 2>(1, "double 2d");
  run_type<double, 2>(2, "double 2d");
  run_type<double, 2>(3, "double 2d");
  run_type<double, 3>(1, "double 3d");
  run_type<double, 3>(2, "double 3d");
  run_type<double, 3>(3, "double 3d");
  run_type<float, 1>(1, "float 1d");
  run_type<float, 1>(2, "float 1d");
  run_type<float, 1>(3, "float 1d");
  run_type<float, 2>(1, "float 2d");
  run_type<float, 2>(2, "float 2d");
  run_type<float, 2>(3, "float 2d");
  run_type<float, 3>(1, "float 3d");
  run_type<float, 3>(2, "float 3d");
  run_type<float, 3>(3, "float 3d");
  run_bend();
  if (fails) {
    std::cout << "cpp_interface: " << fails << " check(s) failed\n";
    return 1;
  }
  std::cout << "cpp_interface: all checks passed\n";
  return 0;
}
