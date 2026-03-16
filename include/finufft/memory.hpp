#pragma once

// Cross-platform RAII wrapper for large temporary buffers.
//
// Uses mmap/VirtualAlloc for large allocations with two key features:
// 1. allocation keeps a stable virtual address range for reuse
// 2. MADV_FREE / MEM_RESET marks pages as reclaimable by the OS without
//    releasing the virtual address range, so subsequent reuse is free
//    unless the OS reclaimed the pages under memory pressure.

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#else
#include <cstring> // memset fallback
#endif

namespace finufft {

// Large-buffer allocator that returns page-aligned memory and supports
// marking pages as reclaimable between uses.
class ReclaimableMemory {
public:
  // Heap allocation alignment for small buffers.
  static constexpr size_t ALIGNMENT = 64;
  // Below this size, use aligned_alloc instead of mmap to avoid page-table overhead.
  static constexpr size_t MIN_MMAP_SIZE = size_t{1} << 18; // 256 KB
  // Below this size, skip madvise: the syscall + TLB flush cost exceeds memory savings.
  static constexpr size_t MIN_RECLAIMABLE = size_t{1} << 20; // 1 MB

  ReclaimableMemory() = default;

  // Non-copyable, movable
  ReclaimableMemory(const ReclaimableMemory &)            = delete;
  ReclaimableMemory &operator=(const ReclaimableMemory &) = delete;
  ReclaimableMemory(ReclaimableMemory &&o) noexcept
      : ptr_(o.ptr_), nbytes_(o.nbytes_), is_mmap_(o.is_mmap_) {
    o.ptr_     = nullptr;
    o.nbytes_  = 0;
    o.is_mmap_ = false;
  }
  ReclaimableMemory &operator=(ReclaimableMemory &&o) noexcept {
    if (this != &o) {
      deallocate();
      ptr_       = o.ptr_;
      nbytes_    = o.nbytes_;
      is_mmap_   = o.is_mmap_;
      o.ptr_     = nullptr;
      o.nbytes_  = 0;
      o.is_mmap_ = false;
    }
    return *this;
  }

  ~ReclaimableMemory() { deallocate(); }

  // Allocate nbytes of memory while leaving physical pages to be faulted in on
  // first use. Returns true on success.
  bool allocate(size_t nbytes) {
    if (nbytes == 0) return true;
    if (ptr_ && nbytes_ >= nbytes) return true; // existing allocation is large enough
    deallocate();
    nbytes_ = nbytes;
#if defined(_WIN32)
    if (nbytes < MIN_MMAP_SIZE) {
      // Small buffers: use aligned heap allocation to avoid page-table overhead
      constexpr size_t align = ALIGNMENT;
      ptr_     = _aligned_malloc(((nbytes + align - 1) / align) * align, align);
      is_mmap_ = false;
      if (!ptr_) {
        nbytes_ = 0;
        return false;
      }
    } else {
      ptr_     = VirtualAlloc(nullptr, nbytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      is_mmap_ = true;
      if (!ptr_) {
        nbytes_ = 0;
        return false;
      }
    }
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
    if (nbytes < MIN_MMAP_SIZE) {
      // Small buffers: use aligned heap allocation to avoid mmap/page-table overhead
      constexpr size_t align = ALIGNMENT;
      ptr_     = std::aligned_alloc(align, ((nbytes + align - 1) / align) * align);
      is_mmap_ = false;
      if (!ptr_) {
        nbytes_ = 0;
        return false;
      }
    } else {
      ptr_ = mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                  -1, 0);
      is_mmap_ = true;
      if (ptr_ == MAP_FAILED) {
        ptr_    = nullptr;
        nbytes_ = 0;
        return false;
      }
    }
#else
    // Fallback: aligned allocation
    constexpr size_t align = ALIGNMENT;
    ptr_     = std::aligned_alloc(align, ((nbytes + align - 1) / align) * align);
    is_mmap_ = false;
    if (!ptr_) {
      nbytes_ = 0;
      return false;
    }
    std::memset(ptr_, 0, nbytes);
#endif
    return true;
  }

  // Mark pages as reclaimable by the OS. The virtual address range is kept,
  // and pages may remain resident if there is no memory pressure.
  // After this call, the contents are undefined until the next write.
  void mark_reclaimable() {
    // Skip for small buffers or non-mmap allocations: the syscall + TLB flush
    // overhead exceeds the memory savings. Pages stay warm in cache for reuse.
    if (!ptr_ || !is_mmap_ || nbytes_ < MIN_RECLAIMABLE) return;
#if defined(_WIN32)
    // MEM_RESET tells Windows the pages are no longer needed.
    // Pages remain committed but can be discarded under pressure.
    VirtualAlloc(ptr_, nbytes_, MEM_RESET, PAGE_READWRITE);
#elif defined(__linux__) || defined(__APPLE__)
    madvise(ptr_, nbytes_, MADV_FREE);
#endif
    // Other platforms: no-op, pages stay resident
  }

  void *data() const { return ptr_; }
  size_t size() const { return nbytes_; }

private:
  void deallocate() {
    if (!ptr_) return;
#if defined(_WIN32)
    if (is_mmap_)
      VirtualFree(ptr_, 0, MEM_RELEASE);
    else
      _aligned_free(ptr_);
#elif defined(__unix__) || defined(__APPLE__)
    if (is_mmap_)
      munmap(ptr_, nbytes_);
    else
      std::free(ptr_);
#else
    std::free(ptr_);
#endif
    ptr_    = nullptr;
    nbytes_ = 0;
  }

  void *ptr_     = nullptr;
  size_t nbytes_ = 0;
  bool is_mmap_  = false;
};

} // namespace finufft
