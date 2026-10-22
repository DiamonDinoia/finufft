import ctypes, os, sys

# Why does the hidden static libomp not raise Error #15 next to Homebrew libomp?
# Print each runtime's banner (KMP_VERSION=1) and the registration variable.
order, brew = sys.argv[1], sys.argv[2]
libc = ctypes.CDLL(None)
libc.getenv.restype = ctypes.c_char_p
name = f"__KMP_REGISTERED_LIB_{os.getpid()}".encode()


def show(tag):
    print(tag, "registration:", libc.getenv(name), flush=True)


def start_brew():
    ctypes.CDLL(brew, mode=ctypes.RTLD_GLOBAL).omp_get_max_threads()
    show("after brew init")


def start_finufft():
    import numpy as np, finufft

    show("after finufft import")
    finufft.nufft1d1(np.zeros(100), np.ones(100, complex), 16, nthreads=2)
    show("after finufft call")


show("start")
for step in (
    (start_brew, start_finufft)
    if order == "brew-first"
    else (start_finufft, start_brew)
):
    step()
print("done", order)
