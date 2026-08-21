#include <finufft/test_defs.hpp>

/* Test that a type-3 plan reused across many setpts calls gives the same answer
   as a fresh plan per point set.

   The type-3 setpts caches target-side work (invPhiHat, STUp, the inner type-2
   plan). STUp depends on the rescale factor gam and invPhiHat on the spreading
   kernel width. A cache key that omits them reuses stale arrays and returns
   O(1) garbage.

   The targets are fixed fermionic Matsubara frequencies, so the target
   coordinates always compare equal and the cache is always consulted. The cache
   holds one slot, so only consecutive calls can collide, and the source box
   therefore grows by a small fixed ratio per call. The ladder spans the two
   regimes that hold nfdim still while the target work moves. Below X = 1/S,
   nhg_type3 clamps X*S and floors nf at 2*nspread, so nf and nspread hold while
   gam tracks X. Above it, an upsampfac flip moves nspread and gam together
   while next235 or the same floor returns nf to its previous value.
*/

using namespace std;

int main() {
  BIGINT M   = 37;  // source pts per call, few enough that the fine grid stays small
  BIGINT N   = 200; // target freqs, fixed across calls
  int ncalls = 240; // geometric ladder of source box widths
  FLT boxlo  = 0.0002; // below 1/S, where nf is pinned at 2*nspread and only gam moves
  FLT boxhi  = 10.0;
  int isign  = +1;
  FLT beta   = 20.0;
#ifdef SINGLE
  FLT tol        = 1e-6;
  FLT allowederr = 1e-4;
  string name    = "type3_replanf";
#else
  FLT tol        = 1e-12;
  FLT allowederr = 1e-9;
  string name    = "type3_replan";
#endif
  cout << name << ": M=" << (long long)M << " N=" << (long long)N << " calls=" << ncalls
       << " tol=" << tol << endl;

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.allow_eps_too_small = 1;
  opts.nthreads            = 1;

  vector<FLT> s(N);
  for (BIGINT k = 0; k < N; ++k) s[k] = (FLT)((2.0 * (k - N / 2) + 1.0) * PI / beta);

  FINUFFT_PLAN reused = NULL;
  int ier             = FINUFFT_MAKEPLAN(3, 1, NULL, isign, 1, tol, &reused, &opts);
  if (ier > 1) {
    cout << name << ": makeplan ier=" << ier << endl;
    return ier;
  }

  vector<FLT> x(M);
  vector<CPX> c(M), f_reused(N), f_fresh(N);
  for (BIGINT j = 0; j < M; ++j) c[j] = CPX(cos(0.7 * j + 1.0), sin(1.3 * j));

  FLT maxerr   = 0;
  int worst    = 0;
  FLT worstbox = 0;

  for (int call = 0; call < ncalls; ++call) {
    // deterministic ladder, so the source half width is exactly box/2 every call
    FLT box = boxlo * pow(boxhi / boxlo, (FLT)call / (ncalls - 1)) * beta;
    for (BIGINT j = 0; j < M; ++j) x[j] = box * (FLT)j / (FLT)(M - 1);

    ier = FINUFFT_SETPTS(reused, M, x.data(), NULL, NULL, N, s.data(), NULL, NULL);
    if (ier > 1) return ier;
    ier = FINUFFT_EXECUTE(reused, c.data(), f_reused.data());
    if (ier > 1) return ier;

    // same point set through a plan that has never been used before
    FINUFFT_PLAN fresh = NULL;
    ier                = FINUFFT_MAKEPLAN(3, 1, NULL, isign, 1, tol, &fresh, &opts);
    if (ier > 1) return ier;
    ier = FINUFFT_SETPTS(fresh, M, x.data(), NULL, NULL, N, s.data(), NULL, NULL);
    if (ier > 1) return ier;
    ier = FINUFFT_EXECUTE(fresh, c.data(), f_fresh.data());
    if (ier > 1) return ier;
    FINUFFT_DESTROY(fresh);

    for (BIGINT k = 0; k < N; ++k) {
      FLT e = abs(f_reused[k] - f_fresh[k]);
      if (e > maxerr) {
        maxerr   = e;
        worst    = call;
        worstbox = box;
      }
    }
  }
  FINUFFT_DESTROY(reused);

  cout << name << ": max |reused - fresh| = " << maxerr << " (worst call " << worst
       << ", box " << worstbox << "), allowed " << allowederr << endl;
  if (maxerr > allowederr) {
    cout << name << ": FAILED, stale type-3 cache" << endl;
    return 1;
  }
  cout << name << ": pass" << endl;
  return 0;
}
