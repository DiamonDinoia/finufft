PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
# make predefines CXX=g++, so ?= would never fire here.
ifeq ($(origin CXX),default)
CXX = c++
endif
CXXFLAGS ?= -std=c++17 -O2

app: main.cpp
	$(CXX) $(CXXFLAGS) -I$(PREFIX)/include $< -o $@ \
		-L$(LIBDIR) -lfinufft -Wl,-rpath,$(LIBDIR)

clean:
	rm -f app

.PHONY: clean
