# Pipelined Tensor Core GEMM

This is a compact CUDA GEMM project focused on half-precision matrix
multiplication with Tensor Cores. The kernel uses shared-memory tiling, async
global-to-shared copies, and MMA instructions to compute `C = A x B`.

The goal is to keep the project small enough to read, but complete enough to
build, validate, and benchmark locally.

## Project Layout

```text
src/        GEMM kernel, public launcher, and benchmark driver
includes/   Low-level MMA helper intrinsics
tests/      Sample matrices, reference checker, and pass/fail validation
```

Generated binaries are written to `build/`.

## Requirements

- NVIDIA GPU
- CUDA toolkit with `nvcc`
- GNU Make
- Bash

The default build targets `sm_89`. If your GPU uses a different CUDA
architecture, pass it through `ARCH`:

```bash
make build ARCH=sm_80
```

Some common values:

```text
sm_75  Turing
sm_80  Ampere A100
sm_86  Ampere RTX 30-series
sm_89  Ada RTX 40-series
```

## Build

```bash
make build
```

This creates two binaries:

```text
build/benchmark
build/validation_runner
```

## Validate Correctness

```bash
make validate
```

Validation runs a set of generated GEMM cases and compares this kernel against a
bundled CUTLASS-based reference checker. It reports pass/fail rather than a
score.

You can also validate one shape directly:

```bash
./build/validation_runner 1024 512 256
```

The current kernel expects `M`, `N`, and `K` to be multiples of 64.

## Run Benchmarks

```bash
make benchmark
```

Benchmarking does two things:

- runs the sample matrix files in `tests/sample_data/` and checks their expected outputs
- runs larger synthetic GEMM sizes and prints average runtime plus TFLOPS

You can run a synthetic benchmark directly:

```bash
./build/benchmark 2048 2048 2048
```

In this mode, the input matrices are filled with `0.1`, and the program reports
timing only.

## Run Custom Inputs

To test your own matrix files:

```bash
./build/benchmark path/to/A.txt path/to/B.txt path/to/C.txt
```

Each text file should start with matrix dimensions:

```text
rows cols
v0 v1 v2 ...
```

For example, if `A` is `128 x 64` and `B` is `64 x 128`, then `C` should be
`128 x 128`:

```bash
./build/benchmark inputs/A_128x64.txt inputs/B_64x128.txt inputs/C_128x128.txt
```

The benchmark loader stores `A` as row-major and converts `B` into column-major
layout before passing it to the kernel.

## Clean

```bash
make clean
```

## Kernel Notes

The public launcher is:

```cpp
void launchGemmKernel(int M, int N, int K, const __half* A, const __half* B, float* C);
```

The kernel computes one `64 x 64` output tile per CTA using 128 threads.

Input/output layout:

- `A`: row-major `__half`
- `B`: column-major `__half`
- `C`: row-major `float`

Internally, the kernel stages two `64 x 64` tiles for `A` and `B` in dynamic
shared memory. A two-stage CUDA pipeline overlaps global-memory copies with MMA
compute so the Tensor Core work can stay fed.
