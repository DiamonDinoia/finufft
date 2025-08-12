#include <array>
#include <intrin.h>
#include <iostream>

static inline void cpuid(int out[4], int leaf, int subleaf = 0) {
  __cpuidex(out, leaf, subleaf);
}

static inline int max_basic_leaf() {
  int r[4]; __cpuid(r, 0);
  return r[0];
}

static inline bool os_avx_enabled() {
  int r[4]; __cpuid(r, 1);
  bool osxsave = (r[2] & (1 << 27)) != 0;     // ECX[27] OSXSAVE
  if (!osxsave) return false;
  unsigned long long xcr0 = _xgetbv(0);       // XCR0
  // Need XMM (bit1) and YMM (bit2)
  return (xcr0 & 0x6) == 0x6;
}

bool is_sse2_supported() {
  int r[4]; __cpuid(r, 1);
  return (r[3] & (1 << 26)) != 0;             // EDX[26]
}

bool is_avx_supported() {
  int r[4]; __cpuid(r, 1);
  bool cpuAVX = (r[2] & (1 << 28)) != 0;      // ECX[28]
  return cpuAVX && os_avx_enabled();
}

bool is_avx2_supported() {
  if (!is_avx_supported()) return false;      // implies OSXSAVE + XCR0 XMM/YMM
  if (max_basic_leaf() < 7) return false;
  int r[4]; cpuid(r, 7, 0);
  return (r[1] & (1 << 5)) != 0;              // EBX[5]
}

bool is_avx512_supported() {
  if (!is_avx_supported()) return false;
  if (max_basic_leaf() < 7) return false;
  int r[4]; cpuid(r, 7, 0);
  bool avx512f = (r[1] & (1 << 16)) != 0;     // EBX[16] AVX-512F
  if (!avx512f) return false;
  unsigned long long xcr0 = _xgetbv(0);
  // Need XMM(bit1), YMM(bit2), Opmask(bit5), ZMM_hi256(bit6), Hi16_ZMM(bit7)
  const unsigned long long mask = (1ull<<1)|(1ull<<2)|(1ull<<5)|(1ull<<6)|(1ull<<7);
  return (xcr0 & mask) == mask;
}

int main() {
  if (is_avx512_supported())      std::cout << "AVX512";
  else if (is_avx2_supported())   std::cout << "AVX2";
  else if (is_avx_supported())    std::cout << "AVX";
  else if (is_sse2_supported())   std::cout << "SSE2";
  else                            std::cout << "NONE";
  return 0;
}
