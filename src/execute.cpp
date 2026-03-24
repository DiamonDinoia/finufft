#include <finufft/execute.hpp>

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

// Suppress re-instantiation of all spread/interp method templates.
extern template int FINUFFT_PLAN_T<FLT>::spreadSorted(FLT *FINUFFT_RESTRICT,
                                                       const FLT *) const;
extern template int FINUFFT_PLAN_T<FLT>::interpSorted(FLT *FINUFFT_RESTRICT,
                                                       FLT *FINUFFT_RESTRICT) const;
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
template int FINUFFT_PLAN_T<FLT>::execute_internal(
    std::complex<FLT> *cj, std::complex<FLT> *fk, bool adjoint, int ntrans_actual,
    std::complex<FLT> *aligned_scratch, size_t scratch_size) const;
