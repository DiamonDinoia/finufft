#ifndef FINUFFT_XSIMD_HPP
#define FINUFFT_XSIMD_HPP

#include <xsimd/xsimd.hpp>

#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
#undef XSIMD_NO_SUPPORTED_ARCHITECTURE
#undef XSIMD_CONFIG_HPP
#undef XSIMD_HPP
#define XSIMD_WITH_EMULATED 1
#define XSIMD_DEFAULT_ARCH  emulated<128>
#include <xsimd/xsimd.hpp>
#endif

#endif // FINUFFT_XSIMD_HPP
