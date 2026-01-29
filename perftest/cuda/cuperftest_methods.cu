#include <cstdint>
#include <getopt.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cufinufft.h>
#include <cufinufft/impl.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using finufft::common::PI;

struct CaseConfig {
  char prec;
  int type;
  int dim;
  int N[3];
  int64_t M;
  int ntransf;
  double tol;
};

struct CudaTimer {
  ~CudaTimer() {
    for (auto &event : start_) cudaEventDestroy(event);
    for (auto &event : stop_) cudaEventDestroy(event);
  }

  void start() {
    start_.push_back(cudaEvent_t{});
    stop_.push_back(cudaEvent_t{});
    cudaEventCreate(&start_.back());
    cudaEventCreate(&stop_.back());
    cudaEventRecord(start_.back());
  }

  void stop() { cudaEventRecord(stop_.back()); }

  void sync() {
    for (auto &event : stop_) cudaEventSynchronize(event);
  }

  float tot() {
    float dt_tot = 0.f;
    for (int i = 0; i < static_cast<int>(start_.size()); ++i) {
      float dt = 0.f;
      cudaEventElapsedTime(&dt, start_[i], stop_[i]);
      dt_tot += dt;
    }
    return dt_tot;
  }

  std::vector<cudaEvent_t> start_;
  std::vector<cudaEvent_t> stop_;
};

void gpu_warmup() {
  int nf1 = 64;
  cufftHandle fftplan;
  cufftPlan1d(&fftplan, nf1, CUFFT_Z2Z, 1);
  thrust::device_vector<cufftDoubleComplex> in(nf1), out(nf1);
  cufftExecZ2Z(fftplan, in.data().get(), out.data().get(), 1);
  cudaDeviceSynchronize();
  cufftDestroy(fftplan);
}

struct TunedCase {
  int N[3];
  int64_t M;
};

double estimate_per_m_bytes(const CaseConfig &cfg, int method);

TunedCase tune_shared_dim3(char prec, int ntransf, double density) {
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  double target_frac = 0.30;
  if (prec == 'd') target_frac = 0.05;
  const double target_bytes = static_cast<double>(free_bytes) * target_frac;

  CaseConfig worst{};
  worst.prec = prec;
  worst.type = 3;
  worst.dim = 3;
  worst.ntransf = ntransf;
  double per_m = estimate_per_m_bytes(worst, 2) * (prec == 'd' ? 3.5 : 1.6);
  const double complex_bytes = prec == 'f' ? 8.0 : 16.0;
  const double grid_per = complex_bytes * ntransf * 1.6;
  const double per_grid = grid_per + density * per_m;

  int64_t target_grid = static_cast<int64_t>(target_bytes / std::max(per_grid, 1.0));
  if (target_grid < 8) target_grid = 8;

  int side = static_cast<int>(std::ceil(std::cbrt(static_cast<double>(target_grid))));
  if (prec == 'd') {
    side = std::min(side, 16);
  }
  int n1 = side;
  int n2 = side;
  int n3 = side;

  for (int iter = 0; iter < 10; ++iter) {
    int64_t ngrid = static_cast<int64_t>(n1) * n2 * n3;
    int64_t M = static_cast<int64_t>(density * ngrid);
    double est = static_cast<double>(ngrid) * grid_per + static_cast<double>(M) * per_m;
    if (est <= target_bytes) break;
    int64_t reduced = static_cast<int64_t>(ngrid * 0.7);
    if (reduced < 8) reduced = 8;
    side = static_cast<int>(std::ceil(std::cbrt(static_cast<double>(reduced))));
    if (prec == 'd') {
      side = std::min(side, 16);
    }
    n1 = side;
    n2 = side;
    n3 = side;
  }

  int64_t ngrid = static_cast<int64_t>(n1) * n2 * n3;
  int64_t M = static_cast<int64_t>(density * ngrid);
  return TunedCase{{n1, n2, n3}, M};
}

double estimate_per_m_bytes(const CaseConfig &cfg, int method) {
  const double real_bytes = cfg.prec == 'f' ? 4.0 : 8.0;
  const double complex_bytes = cfg.prec == 'f' ? 8.0 : 16.0;
  const int64_t ntransf = cfg.ntransf;

  double point_bytes = static_cast<double>(cfg.dim) * real_bytes;
  if (cfg.type == 3) {
    point_bytes *= 2.0; // src + tgt points
  }

  double data_bytes = complex_bytes * ntransf;
  if (cfg.type == 3) {
    data_bytes *= 2.0; // input + output
  }

  double per_m = point_bytes + data_bytes + 16.0; // indices/bookkeeping
  // Extra headroom for internal buffers and temporary workspace.
  double overhead = 3.0;
  if (cfg.type == 3 && method >= 2) {
    overhead = 3.8;
  }
  return per_m * overhead;
}

double estimate_total_bytes(const CaseConfig &cfg, const TunedCase &tc, int method) {
  const double complex_bytes = cfg.prec == 'f' ? 8.0 : 16.0;
  const int64_t ntransf = cfg.ntransf;
  const int64_t ngrid = static_cast<int64_t>(tc.N[0]) * tc.N[1] * tc.N[2];
  const int64_t M = tc.M;

  double grid_bytes = 0.0;
  if (cfg.type == 1 || cfg.type == 2) {
    grid_bytes = static_cast<double>(ngrid) * complex_bytes * ntransf;
  }

  double per_m = estimate_per_m_bytes(cfg, method);
  return grid_bytes + static_cast<double>(M) * per_m;
}

TunedCase tune_case(const CaseConfig &cfg, int method, double density) {
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  double target_frac = 0.40;
  if (cfg.type == 3 && method >= 2) {
    target_frac = 0.25;
  }
  if (cfg.type == 2 && cfg.dim == 2 && method == 3) {
    target_frac = 0.25;
  }
  const double target_bytes = static_cast<double>(free_bytes) * target_frac;

  int n1 = 1;
  int n2 = 1;
  int n3 = 1;
  int64_t ngrid = static_cast<int64_t>(n1) * n2 * n3;
  int64_t M = static_cast<int64_t>(density * ngrid);

  auto update_dims = [&](int64_t target_grid) {
    if (cfg.dim == 1) {
      n1 = static_cast<int>(target_grid);
      n2 = 1;
      n3 = 1;
    } else if (cfg.dim == 2) {
      int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(target_grid))));
      n1 = side;
      n2 = side;
      n3 = 1;
    } else {
      int side = static_cast<int>(std::ceil(std::cbrt(static_cast<double>(target_grid))));
      n1 = side;
      n2 = side;
      n3 = side;
    }
  };

  const double complex_bytes = cfg.prec == 'f' ? 8.0 : 16.0;
  const double per_m = estimate_per_m_bytes(cfg, method) * 1.4;
  const double grid_per =
      (cfg.type == 1 || cfg.type == 2) ? complex_bytes * cfg.ntransf * 1.5 : 0.0;
  const double per_grid = grid_per + density * per_m;
  int64_t target_grid = static_cast<int64_t>(target_bytes / std::max(per_grid, 1.0));
  if (target_grid < 8) target_grid = 8;
  update_dims(target_grid);

  for (int iter = 0; iter < 10; ++iter) {
    ngrid = static_cast<int64_t>(n1) * n2 * n3;
    M = static_cast<int64_t>(density * ngrid);
    TunedCase tc{{n1, n2, n3}, M};
    double est = estimate_total_bytes(cfg, tc, method);
    if (est <= target_bytes) break;
    int64_t reduced = static_cast<int64_t>(ngrid * 0.7);
    if (reduced < 8) reduced = 8;
    update_dims(reduced);
  }

  ngrid = static_cast<int64_t>(n1) * n2 * n3;
  M = static_cast<int64_t>(density * ngrid);
  return TunedCase{{n1, n2, n3}, M};
}

template<typename T>
struct RunResult {
  int method;
  float plan_ms;
  float setpts_ms;
  float exec_ms;
  double setpts_pts_s;
  double exec_pts_s;
};

template<typename T>
RunResult<T> run_case(const CaseConfig &cfg, int method, int n_runs, TunedCase *tuned_out) {
  TunedCase tuned = cfg.M > 0 ? TunedCase{{cfg.N[0], cfg.N[1], cfg.N[2]}, cfg.M}
                              : tune_case(cfg, method, 5.0);
  if (tuned_out) *tuned_out = tuned;
  const int64_t M = tuned.M;
  const int N = tuned.N[0] * tuned.N[1] * tuned.N[2];
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  double est_bytes = estimate_total_bytes(cfg, tuned, method);
  std::cout << "# tuned N=" << tuned.N[0] << "x" << tuned.N[1] << "x" << tuned.N[2]
            << " M=" << M << " density=5"
            << " est=" << est_bytes / (1024.0 * 1024.0 * 1024.0) << "GiB"
            << " free=" << free_bytes / (1024.0 * 1024.0 * 1024.0) << "GiB\n";
  const int type = cfg.type;
  constexpr int iflag = 1;

  thrust::host_vector<T> x(M * cfg.ntransf), y(M * cfg.ntransf), z(M * cfg.ntransf);
  thrust::host_vector<T> s(M * cfg.ntransf), t(M * cfg.ntransf), u(M * cfg.ntransf);
  thrust::host_vector<thrust::complex<T>> c(M * cfg.ntransf);
  thrust::host_vector<thrust::complex<T>> fk((cfg.type == 3 ? M : N) * cfg.ntransf);

  thrust::device_vector<T> d_x(M * cfg.ntransf), d_y(M * cfg.ntransf), d_z(M * cfg.ntransf);
  thrust::device_vector<T> d_s(M * cfg.ntransf), d_t(M * cfg.ntransf), d_u(M * cfg.ntransf);
  thrust::device_vector<thrust::complex<T>> d_c(M * cfg.ntransf);
  thrust::device_vector<thrust::complex<T>> d_fk((cfg.type == 3 ? M : N) * cfg.ntransf);

  std::default_random_engine eng(1);
  std::uniform_real_distribution<T> dist11(-1, 1);
  auto randm11 = [&eng, &dist11]() { return dist11(eng); };

  for (int64_t i = 0; i < M; ++i) {
    x[i] = PI * randm11();
    y[i] = PI * randm11();
    z[i] = PI * randm11();
    s[i] = PI * randm11();
    t[i] = PI * randm11();
    u[i] = PI * randm11();
  }
  for (int64_t i = M; i < M * cfg.ntransf; ++i) {
    int64_t j = i % M;
    x[i] = x[j];
    y[i] = y[j];
    z[i] = z[j];
    s[i] = s[j];
    t[i] = t[j];
    u[i] = u[j];
  }

  if (type == 1 || type == 3) {
    for (int i = 0; i < M * cfg.ntransf; ++i) {
      c[i].real(randm11());
      c[i].imag(randm11());
    }
  } else if (type == 2) {
    for (int i = 0; i < N * cfg.ntransf; ++i) {
      fk[i].real(randm11());
      fk[i].imag(randm11());
    }
  } else {
    std::cerr << "Invalid type " << type << " supplied\n";
    return {};
  }

  gpu_warmup();

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_method = method;

  cufinufft_plan_t<T> *dplan = nullptr;
  CudaTimer plan_timer, setpts_timer, execute_timer;

  d_x = x;
  d_y = y;
  d_z = z;
  d_s = s;
  d_t = t;
  d_u = u;
  if (type == 1 || type == 3) d_c = c;
  if (type == 2) d_fk = fk;

  T *d_x_p = cfg.dim >= 1 ? d_x.data().get() : nullptr;
  T *d_y_p = cfg.dim >= 2 ? d_y.data().get() : nullptr;
  T *d_z_p = cfg.dim == 3 ? d_z.data().get() : nullptr;
  T *d_s_p = cfg.dim >= 1 ? d_s.data().get() : nullptr;
  T *d_t_p = cfg.dim >= 2 ? d_t.data().get() : nullptr;
  T *d_u_p = cfg.dim == 3 ? d_u.data().get() : nullptr;
  cuda_complex<T> *d_c_p = (cuda_complex<T> *)d_c.data().get();
  cuda_complex<T> *d_fk_p = (cuda_complex<T> *)d_fk.data().get();

  int Nlocal[3] = {tuned.N[0], tuned.N[1], tuned.N[2]};
  plan_timer.start();
  cufinufft_makeplan_impl<T>(cfg.type, cfg.dim, Nlocal, iflag, cfg.ntransf, cfg.tol, &dplan, &opts);
  plan_timer.stop();

  for (int i = 0; i < n_runs; ++i) {
    setpts_timer.start();
    if (type == 3) {
      cufinufft_setpts_impl<T>(M, d_x_p, d_y_p, d_z_p, M, d_s_p, d_t_p, d_u_p, dplan);
    } else {
      cufinufft_setpts_impl<T>(M, d_x_p, d_y_p, d_z_p, 0, nullptr, nullptr, nullptr, dplan);
    }
    setpts_timer.stop();

    execute_timer.start();
    cufinufft_execute_impl<T>(d_c_p, d_fk_p, dplan);
    execute_timer.stop();
  }

  plan_timer.sync();
  setpts_timer.sync();
  execute_timer.sync();

  cufinufft_destroy_impl<T>(dplan);

  const int64_t nupts_tot = M * n_runs * cfg.ntransf;
  double setpts_pts_s = (setpts_timer.tot() > 0.0f)
                            ? nupts_tot * 1000.0 / setpts_timer.tot()
                            : 0.0;
  double exec_pts_s = (execute_timer.tot() > 0.0f)
                          ? nupts_tot * 1000.0 / execute_timer.tot()
                          : 0.0;

  return RunResult<T>{
      method,
      plan_timer.tot(),
      setpts_timer.tot(),
      execute_timer.tot(),
      setpts_pts_s,
      exec_pts_s,
  };
}

bool parse_result_line(const std::string &line, RunResult<double> &res, TunedCase &tuned) {
  if (line.rfind("RESULT,", 0) != 0) return false;
  std::stringstream ss(line);
  std::string token;
  std::getline(ss, token, ','); // RESULT
  std::getline(ss, token, ',');
  tuned.N[0] = std::stoi(token);
  std::getline(ss, token, ',');
  tuned.N[1] = std::stoi(token);
  std::getline(ss, token, ',');
  tuned.N[2] = std::stoi(token);
  std::getline(ss, token, ',');
  tuned.M = std::stoll(token);
  std::getline(ss, token, ',');
  res.plan_ms = std::stod(token);
  std::getline(ss, token, ',');
  res.setpts_ms = std::stod(token);
  std::getline(ss, token, ',');
  res.exec_ms = std::stod(token);
  std::getline(ss, token, ',');
  res.setpts_pts_s = std::stod(token);
  std::getline(ss, token, ',');
  res.exec_pts_s = std::stod(token);
  return true;
}

bool run_subprocess(const std::string &cmd, RunResult<double> &out, TunedCase &tuned) {
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) return false;
  char buffer[512];
  bool ok = false;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    std::string line(buffer);
    if (parse_result_line(line, out, tuned)) {
      ok = true;
    }
  }
  int rc = pclose(pipe);
  return ok && rc == 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && std::string(argv[1]) == "--single") {
    if (argc < 13) {
      std::cerr << "Usage: --single prec type dim N1 N2 N3 M ntransf tol method n_runs\n";
      return 2;
    }
    CaseConfig cfg;
    cfg.prec = argv[2][0];
    cfg.type = std::atoi(argv[3]);
    cfg.dim = std::atoi(argv[4]);
    cfg.N[0] = std::atoi(argv[5]);
    cfg.N[1] = std::atoi(argv[6]);
    cfg.N[2] = std::atoi(argv[7]);
    cfg.M = std::atoll(argv[8]);
    cfg.ntransf = std::atoi(argv[9]);
    cfg.tol = std::atof(argv[10]);
    int method = std::atoi(argv[11]);
    int n_runs = std::atoi(argv[12]);
    TunedCase tuned{};
    RunResult<double> res{};
    if (cfg.prec == 'f') {
      auto r = run_case<float>(cfg, method, n_runs, &tuned);
      res.method = r.method;
      res.plan_ms = r.plan_ms;
      res.setpts_ms = r.setpts_ms;
      res.exec_ms = r.exec_ms;
      res.setpts_pts_s = r.setpts_pts_s;
      res.exec_pts_s = r.exec_pts_s;
    } else {
      auto r = run_case<double>(cfg, method, n_runs, &tuned);
      res.method = r.method;
      res.plan_ms = r.plan_ms;
      res.setpts_ms = r.setpts_ms;
      res.exec_ms = r.exec_ms;
      res.setpts_pts_s = r.setpts_pts_s;
      res.exec_pts_s = r.exec_pts_s;
    }
    std::cout << "RESULT," << tuned.N[0] << "," << tuned.N[1] << "," << tuned.N[2] << ","
              << tuned.M << "," << res.plan_ms << "," << res.setpts_ms << "," << res.exec_ms
              << "," << res.setpts_pts_s << "," << res.exec_pts_s << "\n";
    return 0;
  }

  int n_runs = 5;
  if (argc >= 2) {
    n_runs = std::atoi(argv[1]);
    if (n_runs <= 0) n_runs = 5;
  }

  std::vector<CaseConfig> cases = {
      {'f', 1, 1, {10000, 1, 1}, 0, 1, 1e-4},
      {'d', 1, 1, {10000, 1, 1}, 0, 1, 1e-9},
      {'f', 2, 1, {10000, 1, 1}, 0, 1, 1e-4},
      {'d', 2, 1, {10000, 1, 1}, 0, 1, 1e-9},
      {'f', 3, 1, {10000, 1, 1}, 0, 1, 1e-4},
      {'d', 3, 1, {10000, 1, 1}, 0, 1, 1e-9},
      {'f', 1, 2, {256, 256, 1}, 0, 1, 1e-5},
      {'d', 1, 2, {256, 256, 1}, 0, 1, 1e-9},
      {'f', 2, 2, {256, 256, 1}, 0, 1, 1e-5},
      {'d', 2, 2, {256, 256, 1}, 0, 1, 1e-9},
      {'f', 3, 2, {256, 256, 1}, 0, 1, 1e-5},
      {'d', 3, 2, {256, 256, 1}, 0, 1, 1e-9},
      {'f', 1, 3, {24, 24, 16}, 0, 1, 1e-6},
      {'d', 1, 3, {24, 24, 16}, 0, 1, 1e-7},
      {'f', 2, 3, {24, 24, 16}, 0, 1, 1e-6},
      {'d', 2, 3, {24, 24, 16}, 0, 1, 1e-7},
      {'f', 3, 3, {24, 24, 16}, 0, 1, 1e-6},
      {'d', 3, 3, {24, 24, 16}, 0, 1, 1e-7},
  };

  std::cout << "prec,type,dim,N1,N2,N3,M,ntransf,method,"
               "plan_ms,setpts_ms,exec_ms,setpts_pts_s,exec_pts_s,exec_speedup_vs_m1\n";

  TunedCase shared_dim3_f{};
  TunedCase shared_dim3_d{};
  bool shared_dim3_f_set = false;
  bool shared_dim3_d_set = false;

  for (const auto &cfg : cases) {
    std::vector<RunResult<double>> results(3);
    std::vector<TunedCase> tuned(3);
    std::vector<bool> ok(3, false);
    int n1 = cfg.N[0];
    int n2 = cfg.N[1];
    int n3 = cfg.N[2];
    int64_t m_override = 0;
    if (cfg.dim == 3) {
      if (cfg.prec == 'f') {
        if (!shared_dim3_f_set) {
          shared_dim3_f = tune_shared_dim3('f', cfg.ntransf, 5.0);
          shared_dim3_f_set = true;
        }
        n1 = shared_dim3_f.N[0];
        n2 = shared_dim3_f.N[1];
        n3 = shared_dim3_f.N[2];
        m_override = shared_dim3_f.M;
      } else {
        if (!shared_dim3_d_set) {
          shared_dim3_d = tune_shared_dim3('d', cfg.ntransf, 5.0);
          shared_dim3_d_set = true;
        }
        n1 = shared_dim3_d.N[0];
        n2 = shared_dim3_d.N[1];
        n3 = shared_dim3_d.N[2];
        m_override = shared_dim3_d.M;
      }
    }

    bool all_ok = false;
    for (int attempt = 0; attempt < 6; ++attempt) {
      all_ok = true;
      for (int method = 1; method <= 3; ++method) {
        std::stringstream cmd;
        cmd << argv[0] << " --single " << cfg.prec << " " << cfg.type << " " << cfg.dim
            << " " << n1 << " " << n2 << " " << n3 << " " << m_override << " "
            << cfg.ntransf << " " << cfg.tol << " " << method << " " << n_runs
            << " 2>/dev/null";
        ok[method - 1] = run_subprocess(cmd.str(), results[method - 1], tuned[method - 1]);
        results[method - 1].method = method;
        if (!ok[method - 1]) {
          all_ok = false;
        }
      }
      if (all_ok) break;
      n1 = std::max(n1 * 7 / 10, 8);
      n2 = std::max(n2 * 7 / 10, 8);
      n3 = std::max(n3 * 7 / 10, 8);
      int64_t ngrid = static_cast<int64_t>(n1) * n2 * n3;
      m_override = static_cast<int64_t>(5.0 * ngrid);
    }

    if (!all_ok) {
      std::cerr << "ERROR: benchmark failed for prec=" << cfg.prec << " type=" << cfg.type
                << " dim=" << cfg.dim << " after tuning retries; aborting.\n";
      return 1;
    }
    double baseline = ok[0] ? results[0].exec_pts_s : 0.0;
    for (int idx = 0; idx < 3; ++idx) {
      const auto &r = results[idx];
      const auto &t = tuned[idx];
      double speedup = (baseline > 0.0 && ok[idx]) ? r.exec_pts_s / baseline : 0.0;
      double nan = std::numeric_limits<double>::quiet_NaN();
      std::cout << cfg.prec << "," << cfg.type << "," << cfg.dim << ","
                << t.N[0] << "," << t.N[1] << "," << t.N[2] << ","
                << t.M << "," << cfg.ntransf << "," << r.method << ","
                << (ok[idx] ? r.plan_ms : nan) << "," << (ok[idx] ? r.setpts_ms : nan) << ","
                << (ok[idx] ? r.exec_ms : nan) << "," << (ok[idx] ? r.setpts_pts_s : nan) << ","
                << (ok[idx] ? r.exec_pts_s : nan) << "," << speedup << "\n";
    }
  }

  return 0;
}
