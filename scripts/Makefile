
# the suffix to add to our .so file
EXT_SUFFIX=$(shell python3-config --extension-suffix)
PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)

UNAME := $(shell uname)

EXTRA_PYBIND_FLAG = 
ifeq ($(UNAME), Darwin)
# Add -undefined dynamic_lookup flag to compile options
# See special instructions for building on MACOSX for pybind11
    EXTRA_PYBIND_FLAG = -undefined dynamic_lookup
endif

.PHONY: clean all tests

SO_FILES := $(patsubst %.cpp,%${EXT_SUFFIX},$(wildcard *.cpp))

all: $(SO_FILES)
	@:

%$(EXT_SUFFIX) : %.cpp
	c++ -O3 -Wall -shared -std=c++11 -fPIC \
			$(EXTRA_PYBIND_FLAG) \
			$(PYBIND_INCLUDES) \
			$< \
			-o $@

clean:
	rm -f *.so

tests:
	python3 unit_tests.py -v

print-%  : ; @echo $* = $($*)
