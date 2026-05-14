#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "gemm_kernel.h"
#include "../includes/mma_intrinsics.cuh"

namespace cg = cooperative_groups;

__global__ void pipe_gemm_kernel(int M, int N, int K, const __half* A, const __half* B, float* C) {
    int tid = threadIdx.x;
    int blockRow = blockIdx.y * 64;
     int blockCol = blockIdx.x * 64;
    int batch_sz = K / 64;

    extern __shared__ __align__(128) int8_t smem[];

    __half* A_smem = reinterpret_cast<__half*>(smem);
    __half* B_smem = A_smem + 2 * 64 * 64;
    float* C_smem =  reinterpret_cast<float*>(B_smem + 2 * 64 * 64);

    for (int i = tid; i < 64 * 64; i += 128) {
        C_smem[i] = 0.0f;
    }
    __syncthreads();

    cg::thread_block cta = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
    auto pipeline = cuda::make_pipeline(cta, &pipe_state);

    pipeline.producer_acquire();
    for (int copy = tid; copy < 512; copy += 128) {
        int row = copy / 8;
         int col = (copy % 8) * 8;

        cuda::memcpy_async(&A_smem[row * 64 + col], &A[(blockRow + row) * K + col], 16, pipeline);

        cuda::memcpy_async(&B_smem[row * 64 + col], &B[(blockCol + row) * K + col],16,pipeline);
    }
    pipeline.producer_commit();

    for (int batch = 1; batch < batch_sz; batch++) {
        int compute_stage_idx = (batch - 1) % 2;
        int copy_stage_idx = batch % 2;
         int k_start = batch * 64;

        pipeline.producer_acquire();
        for (int copy = tid; copy < 512; copy += 128) {
             int row = copy / 8;
            int col = (copy % 8) * 8;

            cuda::memcpy_async(&A_smem[copy_stage_idx * 4096 + row * 64 + col],&A[(blockRow + row) * K + k_start + col],
                16,pipeline);

            cuda::memcpy_async(&B_smem[copy_stage_idx * 4096 + row * 64 + col],&B[(blockCol + row) * K + k_start + col],
                16,pipeline);
        }
        pipeline.producer_commit();

        pipeline.consumer_wait();

        mma_m16n8k16_f16_f16_smem_row_col_64x64(&A_smem[compute_stage_idx * 4096],&B_smem[compute_stage_idx * 4096],C_smem);

        __syncthreads();
        pipeline.consumer_release();
    }

    pipeline.consumer_wait();
    int compute_stage_idx = (batch_sz - 1) % 2;

    mma_m16n8k16_f16_f16_smem_row_col_64x64(&A_smem[compute_stage_idx * 4096],&B_smem[compute_stage_idx * 4096],C_smem);

    __syncthreads();
    pipeline.consumer_release();

    for (int i = tid; i < 64 * 64; i += 128) {
        int row = i / 64;
        int col = i % 64;
        C[(blockRow + row) * N + (blockCol + col)] = C_smem[i];
    }
}

void launchGemmKernel(int M, int N, int K, const __half* A, const __half* B, float* C) {
    dim3 cta(128, 1, 1);
    dim3 grid(N / 64, M / 64, 1);
    int smemBytes = 65536;

    cudaFuncSetAttribute(pipe_gemm_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smemBytes);

    pipe_gemm_kernel<<<grid, cta, smemBytes>>>(M, N, K, A, B, C);
}

extern "C++" void launchStudentKernel(int M, int N, int K, __half* A, __half* B, float* C) {
    launchGemmKernel(M, N, K, A, B, C);
}
