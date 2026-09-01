CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2
PKGCONFIG ?= pkg-config
STATIC ?=

app: main.cpp
	$(CXX) $(CXXFLAGS) $(shell $(PKGCONFIG) --cflags finufft) $< -o $@ \
		$(shell $(PKGCONFIG) $(STATIC) --libs finufft)

clean:
	rm -f app

.PHONY: clean
