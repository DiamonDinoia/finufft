# Quick start without CMake: build against an installed FINUFFT.
#
#   make PREFIX=<install prefix>
#   ./app
#
# This is the route a hand-written build, a ctypes load or a Julia ccall ends up
# taking, and the only one that reads the installed headers and library without
# the exported CMake target in between.
#
# A shared install is assumed. A static libfinufft leaves its FFT and OpenMP
# dependencies to whoever links it, and finufftTargets.cmake is the only thing
# that knows what those are, so a static link belongs to the CMake routes beside
# this one. rpath is what lets ./app run without LD_LIBRARY_PATH set.
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2

app: ../main.cpp
	$(CXX) $(CXXFLAGS) -I$(PREFIX)/include $< -o $@ \
		-L$(LIBDIR) -lfinufft -Wl,-rpath,$(LIBDIR)

clean:
	rm -f app

.PHONY: clean
