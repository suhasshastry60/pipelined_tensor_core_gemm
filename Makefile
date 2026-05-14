NVCC ?= nvcc
ARCH ?= sm_89
CUDAFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)

BUILD_DIR := build
KERNEL_SRC := src/gemm_kernel.cu
KERNEL_HDR := src/gemm_kernel.h includes/mma_intrinsics.cuh
REFERENCE_LIB := tests/reference/libreference_gemm_testbench.a

.PHONY: build benchmark validate test clean

build: $(BUILD_DIR)/benchmark $(BUILD_DIR)/validation_runner

$(BUILD_DIR)/benchmark: src/benchmark.cu $(KERNEL_SRC) $(KERNEL_HDR)
	mkdir -p $(@D)
	$(NVCC) $(CUDAFLAGS) src/benchmark.cu $(KERNEL_SRC) -o $@

$(BUILD_DIR)/validation_runner: $(KERNEL_SRC) $(KERNEL_HDR) $(REFERENCE_LIB)
	mkdir -p $(@D)
	$(NVCC) $(CUDAFLAGS) $(KERNEL_SRC) $(REFERENCE_LIB) -o $@

benchmark: $(BUILD_DIR)/benchmark
	./build/benchmark tests/sample_data/sample_0/A_64x64.txt tests/sample_data/sample_0/B_64x64.txt tests/sample_data/sample_0/C_64x64.txt
	./build/benchmark tests/sample_data/sample_1/A_128x128.txt tests/sample_data/sample_1/B_128x128.txt tests/sample_data/sample_1/C_128x128.txt
	./build/benchmark tests/sample_data/sample_2/A_128x64.txt tests/sample_data/sample_2/B_64x64.txt tests/sample_data/sample_2/C_128x64.txt
	./build/benchmark 256 256 256
	./build/benchmark 512 512 512
	./build/benchmark 1024 1024 1024
	./build/benchmark 2048 2048 2048
	./build/benchmark 4096 4096 4096

validate: $(BUILD_DIR)/validation_runner
	tests/validate.sh

test: validate

clean:
	rm -rf $(BUILD_DIR)
