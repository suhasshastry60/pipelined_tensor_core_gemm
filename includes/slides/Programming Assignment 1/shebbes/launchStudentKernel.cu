#include <cuda.h>

__global__ void matmul(int M, int N, int K, 
                       int layoutA, int layoutB, 
                       float* A, float* B, float* C) {
    
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < M && col_idx < N) {
        float partial_sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float valA, valB;
            
            // A is M x K
            if (layoutA == 0) 
                valA = A[row_idx * K + k];     // Row-Major
            else              
                valA = A[k * M + row_idx];     // Col-Major

            // B is K x N
            if (layoutB == 0) 
                valB = B[k * N + col_idx];     // Row-Major
            else              
                valB = B[col_idx * K + k];     // Col-Major

            partial_sum += valA * valB;
        }
        C[row_idx * N + col_idx] = partial_sum;
    }
}

void launchStudentKernel(int M, int N, int K, int layoutA,
                         int layoutB,
                         float* A,
                         float* B,
                         float* C) {

    dim3 blockSize(16, 16);
    
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);

    matmul<<<gridSize, blockSize>>>(M, N, K, layoutA, layoutB, A, B, C);
}