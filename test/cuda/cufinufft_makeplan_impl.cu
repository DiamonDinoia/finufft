#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <finufft.h>
#include <finufft/finufft_core.h>
struct finufftf_plan_s : public FINUFFT_PLAN_T<float> {};
struct finufft_plan_s : public FINUFFT_PLAN_T<double> {};
#include <cufinufft.h>
#include <cufinufft/impl.h>
#include <cufinufft/utils.h>

int main() {
  // defaults. tests should shadow them to override
  const int iflag      = 1;
  const float tol      = 1e-4;
  const int ntransf    = 1;
  const int dim        = 3;
  int N[3]             = {10, 20, 15};
  const auto upsampfac = 1.25;
  finufft_opts fin_opts;
  finufft_default_opts(&fin_opts);

  fin_opts.upsampfac = upsampfac;
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.upsampfac = upsampfac;

  static const auto cpu_planer = [iflag, tol, ntransf, dim, N, &fin_opts](
                                     const auto type) {
    int64_t Nl[3] = {int64_t(N[0]), int64_t(N[1]), int64_t(N[2])};
    finufft_plan_s *plan{nullptr};
    assert(finufft_makeplan(type, dim, Nl, iflag, ntransf, tol, &plan, &fin_opts) == 0);
    return plan;
  };

  static const auto cpuf_planer = [iflag, tol, ntransf, dim, N, &fin_opts](
                                      const auto type) {
    int64_t Nl[3] = {int64_t(N[0]), int64_t(N[1]), int64_t(N[2])};
    finufftf_plan_s *plan{nullptr};
    assert(finufftf_makeplan(type, dim, Nl, iflag, ntransf, tol, &plan, &fin_opts) == 0);
    return plan;
  };

  const auto test_type1 = [iflag, tol, ntransf, dim, N, &opts](auto *plan) {
    // plan is a pointer to a type that contains real_t
    using T        = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type = 1;
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)N, iflag, ntransf, T(tol), &plan,
                                      &opts) == 0);
    const auto cpu_plan = [type] {
      if constexpr (std::is_same_v<T, double>) {
        return cpu_planer(type);
      } else {
        return cpuf_planer(type);
      }
    }();
    cudaDeviceSynchronize();
    assert(plan->ms == N[0]);
    assert(plan->mt == N[1]);
    assert(plan->mu == N[2]);
    assert(plan->nf1 >= N[0]);
    assert(plan->nf2 >= N[1]);
    assert(plan->nf3 >= N[2]);
    assert(plan->fftplan != 0);
    assert(plan->fwkerhalf1 != nullptr);
    assert(plan->fwkerhalf2 != nullptr);
    assert(plan->fwkerhalf3 != nullptr);
    assert(plan->spopts.spread_direction == type);
    assert(plan->type == type);
    assert(plan->nf1 == cpu_plan->nfdim[0]);
    assert(plan->nf2 == cpu_plan->nfdim[1]);
    assert(plan->nf3 == cpu_plan->nfdim[2]);
    int nf[]       = {plan->nf1, plan->nf2, plan->nf3};
    T *fwkerhalf[] = {plan->fwkerhalf1, plan->fwkerhalf2, plan->fwkerhalf3};
    for (int idx = 0; idx < dim; ++idx) {
      const auto size = (nf[idx] / 2 + 1);
      std::vector<T> fwkerhalf_host(size, -1);
      const auto ier = cudaMemcpy(fwkerhalf_host.data(), fwkerhalf[idx], size * sizeof(T),
                                  cudaMemcpyDeviceToHost);
      assert(ier == cudaSuccess);
      for (int i = 0; i < size; i++) {
        std::cout << "fwkerhalf[" << i << "]: " << fwkerhalf_host[i] << std::endl;
        std::cout << "phiHat[" << i << "]: " << cpu_plan->phiHat[idx][i] << std::endl;
        std::cout << "eps: " << abs(1 - fwkerhalf_host[i] / cpu_plan->phiHat[idx][i])
                  << std::endl;
        assert(abs(1 - fwkerhalf_host[i] / cpu_plan->phiHat[idx][i]) < tol);
      }
    }
    assert(cufinufft_destroy_impl<T>(plan) == 0);
    if constexpr (std::is_same_v<T, double>) {
      assert(finufft_destroy(cpu_plan) == 0);
    } else {
      assert(finufftf_destroy(cpu_plan) == 0);
    }
    plan = nullptr;
  };
  auto test_type2 = [iflag, tol, ntransf, dim, N, &opts](auto plan) {
    // plan is a pointer to a type that contains real_t
    using T        = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type = 2;
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)N, iflag, ntransf, T(tol), &plan,
                                      &opts) == 0);
    const auto cpu_plan = [type] {
      if constexpr (std::is_same_v<T, double>) {
        return cpu_planer(type);
      } else {
        return cpuf_planer(type);
      }
    }();
    cudaDeviceSynchronize();
    assert(plan->ms == N[0]);
    assert(plan->mt == N[1]);
    assert(plan->mu == N[2]);
    assert(plan->nf1 >= N[0]);
    assert(plan->nf2 >= N[1]);
    assert(plan->nf3 >= N[2]);
    assert(plan->fftplan != 0);
    assert(plan->fwkerhalf1 != nullptr);
    assert(plan->fwkerhalf2 != nullptr);
    assert(plan->fwkerhalf3 != nullptr);
    assert(plan->spopts.spread_direction == type);
    assert(plan->type == type);
    assert(plan->opts.gpu_method == 1);
    assert(plan->nf1 == cpu_plan->nfdim[0]);
    assert(plan->nf2 == cpu_plan->nfdim[1]);
    assert(plan->nf3 == cpu_plan->nfdim[2]);
    assert(plan->spopts.nspread == cpu_plan->spopts.nspread);
    int nf[]       = {plan->nf1, plan->nf2, plan->nf3};
    T *fwkerhalf[] = {plan->fwkerhalf1, plan->fwkerhalf2, plan->fwkerhalf3};
    T *phiHat[]    = {cpu_plan->phiHat[0].data(), cpu_plan->phiHat[1].data(),
                      cpu_plan->phiHat[2].data()};
    for (int idx = 0; idx < dim; ++idx) {
      const auto size = (nf[idx] / 2 + 1);
      std::vector<T> fwkerhalf_host(size, -1);
      const auto ier = cudaMemcpy(fwkerhalf_host.data(), fwkerhalf[idx], size * sizeof(T),
                                  cudaMemcpyDeviceToHost);
      if (ier != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(ier) << std::endl;
      }
      assert(ier == cudaSuccess);
      cudaDeviceSynchronize();
      for (int i = 0; i < size; i++) {
        assert(abs(1 - fwkerhalf_host[i] / phiHat[idx][i]) < tol);
      }
    }
    assert(cufinufft_destroy_impl<T>(plan) == 0);
    cudaDeviceSynchronize();
    if constexpr (std::is_same_v<T, double>) {
      assert(finufft_destroy(cpu_plan) == 0);
    } else {
      assert(finufftf_destroy(cpu_plan) == 0);
    }
    plan = nullptr;
  };
  auto test_type3 = [iflag, tol, ntransf, dim, N, &opts](auto plan) {
    // plan is a pointer to a type that contains real_t
    using T        = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type = 3;
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)N, iflag, ntransf, T(tol), &plan,
                                      &opts) == 0);
    cudaDeviceSynchronize();
    assert(plan->ms == 0);
    assert(plan->mt == 0);
    assert(plan->mu == 0);
    assert(plan->nf1 == 1);
    assert(plan->nf2 == 1);
    assert(plan->nf3 == 1);
    assert(plan->fftplan == 0);
    assert(plan->fwkerhalf1 == nullptr);
    assert(plan->fwkerhalf2 == nullptr);
    assert(plan->fwkerhalf3 == nullptr);
    assert(plan->spopts.spread_direction == type);
    assert(plan->type == type);
    assert(plan->opts.upsampfac == 1.25);
    assert(cufinufft_destroy_impl<T>(plan) == 0);
    plan = nullptr;
    cudaDeviceSynchronize();
  };
  // testing correctness of the plan creation
  cufinufft_plan_t<float> *single_plan{nullptr};
  test_type1(single_plan);
  test_type2(single_plan);
  test_type3(single_plan);
  cufinufft_plan_t<double> *double_plan{nullptr};
  test_type1(double_plan);
  test_type2(double_plan);
  test_type3(double_plan);
  return 0;
}
