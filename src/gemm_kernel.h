#pragma once

#include <cuda_fp16.h>

void launchGemmKernel(int M, int N, int K, const __half* A, const __half* B, float* C);
