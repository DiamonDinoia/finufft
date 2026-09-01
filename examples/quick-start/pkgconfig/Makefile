# make predefines CXX=g++, so ?= would never fire here.
ifeq ($(origin CXX),default)
CXX = c++
endif
CXXFLAGS ?= -std=c++17 -O2
PKGCONFIG ?= pkg-config
STATIC ?=

app: main.cpp
	$(CXX) $(CXXFLAGS) $(shell $(PKGCONFIG) --cflags finufft) $< -o $@ \
		$(shell $(PKGCONFIG) $(STATIC) --libs finufft) \
		-Wl,-rpath,$(shell $(PKGCONFIG) --variable=libdir finufft)

clean:
	rm -f app

.PHONY: clean
