// MATLAB ships an OpenMP runtime older than LLVM 19, and the MEX links that copy
// (see matlab/CMakeLists.txt). clang >= 19 emits a call to __kmpc_dispatch_deinit
// after every dynamic-schedule loop, a symbol that runtime does not export.
// Upstream defines it as an empty function, so this stub is behaviour-identical:
//   void __kmpc_dispatch_deinit(ident_t *loc, kmp_int32 gtid) {}
//   llvm-project/openmp/runtime/src/kmp_dispatch.cpp (LLVM 19 through 21)
// CMake compiles this file into the MEX only where the linked runtime lacks the
// symbol, so a future MATLAB with LLVM >= 19 drops the stub.
extern "C" void __kmpc_dispatch_deinit(void *, int) {}
