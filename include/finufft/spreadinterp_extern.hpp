#pragma once
// Extern template declarations for per-dimension spread/interp entry points.
// These are explicitly instantiated in spreadinterp_1d/2d/3d.cpp (one per dim per
// precision). Include this header in any TU that includes spreadinterp.hpp but should
// NOT re-instantiate these symbols (e.g., execute.cpp, spreadinterp.cpp).

// FLT must be defined (float or double) before including this header.

extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_dispatch<1>(
    BIGINT, BIGINT, BIGINT, UBIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT, UBIGINT,
    const FLT *, const FLT *, const FLT *, const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_dispatch<2>(
    BIGINT, BIGINT, BIGINT, UBIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT, UBIGINT,
    const FLT *, const FLT *, const FLT *, const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_dispatch<3>(
    BIGINT, BIGINT, BIGINT, UBIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT, UBIGINT,
    const FLT *, const FLT *, const FLT *, const FLT *) const noexcept;
extern template int FINUFFT_PLAN_T<FLT>::interpSorted_dispatch<1>(FLT *, FLT *) const;
extern template int FINUFFT_PLAN_T<FLT>::interpSorted_dispatch<2>(FLT *, FLT *) const;
extern template int FINUFFT_PLAN_T<FLT>::interpSorted_dispatch<3>(FLT *, FLT *) const;
