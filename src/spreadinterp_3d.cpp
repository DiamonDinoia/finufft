#include <finufft/spreadinterp.hpp>

// Per-dimension TU: explicit instantiation of 3D entry points for one precision.
// Compiled twice (with/without FINUFFT_SINGLE) to cover both float and double.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

template void FINUFFT_PLAN_T<FLT>::spread_subproblem_dispatch_3d(
    BIGINT, BIGINT, BIGINT, UBIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT, UBIGINT,
    const FLT *, const FLT *, const FLT *, const FLT *) const noexcept;
template int FINUFFT_PLAN_T<FLT>::interpSorted_dispatch<3>(FLT *, FLT *) const;
