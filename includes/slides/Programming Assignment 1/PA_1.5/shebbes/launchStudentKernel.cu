#include <cuda.h>
#include <cuda_fp16.h>
#include "mma_intrinsics.cuh"
// TODO: Implement this function...
// This is for grading. You will use this function with the testbench we provide.
// You can add more functions etc. here if you want.
// You only need to launch your kernel inside this function, everything else will be managed by the testbench.
// M, N, K are matrix dimensions
// A is row major, B is column major, C is row major
// A, B and C are pointers to the matrices

__global__ void matmulTensorCore(int M, int N, int K,
                                 __half* A, __half* B, float* C) {
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int lane = tid % 32;

    int warpRow = warpId / 4;   // 0 or 1
     int warpCol = warpId % 4;   // 0 to 3

    int blockRow = blockIdx.y * 32;
    int blockCol = blockIdx.x * 32;

    int tileRow= blockRow + warpRow * 16;
    int tileCol = blockCol + warpCol * 8;

     __shared__ __half A_shar[8][16 * 16];
    __shared__ __half B_shar[8][16 * 8];
    __shared__ float C_shar[8][16 * 8];

    for (int i = lane; i < 16 * 8; i += 32) {
        C_shar[warpId][i] = 0.0f;
    }
     __syncwarp();

    for (int k = 0; k < K; k += 16) {
        for (int i = lane; i < 16 * 16; i += 32) {
            int r =i / 16;
            int c = i % 16;

            int gr = tileRow + r;
            int gc = k + c;

            if (gr <M && gc < K)
                A_shar[warpId][r * 16 + c] =  A[gr * K + gc];
            else
                A_shar[warpId][r * 16 + c] = __float2half(0.0f);
        }

        for (int i = lane; i < 16 * 8; i += 32) {
            int r = i % 16;
            int c = i / 16;

            int gr = k + r;
            int gc = tileCol + c;

            if (gr < K && gc < N)
                B_shar[warpId][c * 16 +r] = B[gc * K + gr];
            else
                B_shar[warpId][c * 16 + r] = __float2half(0.0f);
        }

        __syncwarp();

        mma_m16n8k16_f16_f16_smem_row_col(A_shar[warpId],B_shar[warpId], C_shar[warpId], 16, 16, 8);

        __syncwarp();
    }

    for (int i = lane; i < 16 * 8; i += 32) {
        int r = i / 8;
         int c = i % 8;

        int gr = tileRow + r;
        int gc = tileCol + c;

         if (gr < M && gc < N)
            C[gr * N + gc] = C_shar[warpId][r * 8 + c];
    }
}

void launchStudentKernel(int M, int N, int K, __half* A,
                                              __half* B,
                                              float* C) {

  // Launch your kernel here with appropriate grid and block sizes...
    dim3 blockSize(256);   // 8 warps
     dim3 gridSize((N + 31) / 32, (M + 31) / 32);

    matmulTensorCore<<<gridSize, blockSize>>>(M, N, K, A, B, C);

}
