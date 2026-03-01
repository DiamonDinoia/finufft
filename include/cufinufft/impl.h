#ifndef CUFINUFFT_IMPL_H
#define CUFINUFFT_IMPL_H

#include <cufinufft/types.h>

template<typename T>
int cufinufft_makeplan_impl(int type, int dim, const int *nmodes, int iflag, int ntransf,
                            T tol, cufinufft_plan_t<T> **d_plan_ptr,
                            const cufinufft_opts *opts);

template<typename T> void cufinufft_destroy_impl(cufinufft_plan_t<T> *d_plan);

#endif
