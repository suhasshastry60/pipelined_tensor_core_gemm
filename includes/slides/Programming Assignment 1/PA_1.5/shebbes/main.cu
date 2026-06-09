#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mma_intrinsics.cuh"

using namespace std;


__global__ void matmulBaseline(int M, int N, int K,
                               __half* A, __half* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
         for (int k = 0; k < K; k++) {
            float a = __half2float(A[row * K + k]);
             float b = __half2float(B[col * K + k]);   // B is column-major
            sum +=a * b;
        }
        C[row * N + col] = sum;
    }
}


__global__ void matmulTensorCore(int M, int N, int K,
                                 __half* A, __half* B, float* C) {
    int tid = threadIdx.x;
    int warpId = tid / 32;
 int lane = tid % 32;
 
 //For 128 threads per block
    // int warpRow = warpId / 2;   // 0 or 1
    //  int warpCol = warpId % 2;   // 0 or 1

     //For 256 and 512 threads per block
    int warpRow = warpId / 4;   // 0,1
     int warpCol = warpId % 4;   // 0,1,2,3

     //For 1024 threads per block
    // int warpRow = warpId / 8;   // 0 to 7
    // int warpCol = warpId % 8;   // 0 to 7

     //For 128 threads per block
    // int blockRow = blockIdx.y * 32;
    // int blockCol = blockIdx.x * 16;


     //For 256 threads per block
    // int blockRow = blockIdx.y * 32;
    //  int blockCol = blockIdx.x * 32;

     //For 512 threads per block
    int blockRow = blockIdx.y * 64;
    int blockCol = blockIdx.x * 32;

    //For 1024 threads per block
    // int blockRow = blockIdx.y * 64;
    // int blockCol = blockIdx.x * 64;    

    int tileRow =blockRow + warpRow * 16;
    int tileCol = blockCol + warpCol * 8;

    //For 128, 256 and 512 threads per block
    // __shared__ __half A_shar[8][16 * 16];
    // __shared__ __half B_shar[8][16 * 8];
    // __shared__ float C_shar[8][16 * 8];

    //For 512 threads per block
    __shared__ __half A_shar[16][16 * 16];
    __shared__ __half B_shar[16][16 * 8];
    __shared__ float C_shar[16][16 * 8];    

    //For 1024 threads per block
    // __shared__ __half A_shar[32][16 * 16];
    // __shared__ __half B_shar[32][16 * 8];
    // __shared__ float C_shar[32][16 * 8];    

    for (int i = lane; i < 16 * 8; i += 32) {
        C_shar[warpId][i] = 0.0f;
    }
    __syncwarp();

    for (int k = 0; k < K; k += 16) {
        for (int i = lane; i < 16 * 16; i += 32) {
            int r = i / 16;
            int c = i % 16;
            A_shar[warpId][r * 16 + c] = A[(tileRow + r) * K + (k + c)];
        }

        for (int i = lane; i < 16 * 8; i += 32) {
            int r = i % 16;
            int c = i / 16;
            B_shar[warpId][c * 16 + r] = B[(tileCol + c) * K + (k + r)];
        }

        __syncwarp();

        mma_m16n8k16_f16_f16_smem_row_col(A_shar[warpId], B_shar[warpId], C_shar[warpId], 16, 16, 8);

        __syncwarp();
    }

    for (int i = lane; i < 16 * 8; i += 32) {
        int r = i / 8;
        int c = i % 8;
        C[(tileRow + r) * N + (tileCol + c)] = C_shar[warpId][r * 8 + c];
    }
}

void launchStudentKernel(int M, int N, int K,
                         __half* A,
                         __half* B,
                         float* C) {

    // dim3 blockSize(128);
    //  dim3 gridSize((N + 15) / 16, (M + 31) / 32);

    // dim3 blockSize(256);
    //  dim3 gridSize(N / 32, M / 32);

    dim3 blockSize(512);
   dim3 gridSize((N + 31) / 32, (M + 63) / 64);

//    dim3 blockSize(1024);
//    dim3 gridSize((N + 63) / 64, (M + 63) / 64);

    matmulTensorCore<<<gridSize, blockSize>>>(M, N, K, A, B, C);
}

void initRandomA(__half* A, int M, int K) {
    for (int i = 0; i < M * K; i++) {
        float x = static_cast<float>(rand()) / RAND_MAX;
        A[i] = __float2half(x);
    }
}

void initRandomB(__half* B, int K, int N) {
    for (int c = 0; c < N; c++) {
        for (int r = 0; r < K; r++) {
            float x = static_cast<float>(rand()) / RAND_MAX;
            B[c * K + r] = __float2half(x);
        }
    }
}

void readMatrixFileHalfRowMajor(const char* filename, __half* data, int rows, int cols) {
    ifstream file(filename);

    int fileRows, fileCols;
     file >> fileRows >> fileCols;

    for (int i = 0; i < rows * cols; i++) {
        float val;
        file >> val;
        data[i] = __float2half(val);
    }

    file.close();
}

void readMatrixFileHalfToColumnMajor(const char* filename, __half* data, int rows, int cols) {
    ifstream file(filename);

    int fileRows, fileCols;
    file >> fileRows >> fileCols;

    vector<float> temp(rows * cols);
     for (int i = 0; i < rows * cols; i++) {
        file >> temp[i];
    }

    file.close();

    for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
           data[c * rows + r] = __float2half(temp[r * cols + c]);
        }
    }
}

void readMatrixFileFloat(const char* filename, float* data, int rows, int cols) {
    ifstream file(filename);

    int fileRows, fileCols;
    file >> fileRows >> fileCols;

     for (int i = 0; i < rows * cols; i++) {
        file >> data[i];
    }

    file.close();
}

void checkResult(float* out, float* ref, int size, const char* name) {
    for (int i = 0; i < size; i++) {
        if (fabs(out[i] - ref[i]) > 1e-2) {
             cout << "[FAIL] " << name << " mismatch at index " << i
                 << "  Calc = " << out[i]
                 << "  Ref = " << ref[i] << endl;
            return;
        }
    }
    cout << "[SUCCESS] " << name << " matches reference" << endl;
}


float run_baseline(int M, int N, int K, __half* h_A, __half* h_B, float* h_C) {
    size_t bytesA = M * K * sizeof(__half);
     size_t bytesB = K * N *sizeof(__half);
    size_t bytesC = M * N * sizeof(float);

    __half *d_A, *d_B;
    float *d_C;

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB,cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytesC);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulBaseline<<<gridSize, blockSize>>>(M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
     cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C,bytesC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
     cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}


float run_tensorcore(int M, int N, int K, __half* h_A, __half* h_B, float* h_C) {
    size_t bytesA = M * K * sizeof(__half);
    size_t bytesB = K * N * sizeof(__half);
    size_t bytesC = M * N * sizeof(float);

    __half *d_A, *d_B;
    float *d_C;

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytesC);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launchStudentKernel(M,N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
     cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

void run_explicit(int M, int N, int K,
                  const char* fileA = nullptr,
                  const char* fileB = nullptr,
                  const char* fileC = nullptr) {

    size_t bytesA = M * K * sizeof(__half);
    size_t bytesB = K * N * sizeof(__half);
    size_t bytesC = M * N * sizeof(float);

    __half* h_A = (__half*)malloc(bytesA);
    __half* h_B = (__half*)malloc(bytesB);
    float* h_C_baseline = (float*)malloc(bytesC);
    float* h_C_tensor = (float*)malloc(bytesC);

    if (fileA && fileB) {
        cout << "Reading input matrices from files..." << endl;
        readMatrixFileHalfRowMajor(fileA, h_A, M, K);
        readMatrixFileHalfToColumnMajor(fileB, h_B, K, N);
    } else {
        cout << "Initializing random input matrices..." << endl;
        initRandomA(h_A, M, K);
        initRandomB(h_B, K, N);
    }

    float baseline_ms = run_baseline(M, N, K, h_A, h_B, h_C_baseline);
    float tensor_ms = run_tensorcore(M, N, K, h_A, h_B, h_C_tensor);

    cout << "Baseline kernel time: " << fixed << setprecision(3) << baseline_ms << " ms" << endl;
    cout << "Tensor core kernel time: " << fixed << setprecision(3) << tensor_ms << " ms" << endl;
     cout << "Speedup: " << fixed << setprecision(3) << (baseline_ms / tensor_ms) << "x" << endl;

    if (fileC) {
        vector<float> ref(M * N);
        readMatrixFileFloat(fileC, ref.data(), M, N);
        checkResult(h_C_baseline, ref.data(), M * N, "Baseline");
        checkResult(h_C_tensor, ref.data(), M * N, "Tensor core");
    }

    free(h_A);
    free(h_B);
    free(h_C_baseline);
    free(h_C_tensor);
}


int main(int argc, char* argv[]) {
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    const char* fileA = (argc >= 5) ? argv[4] : nullptr;
    const char* fileB = (argc >= 6) ? argv[5] : nullptr;
     const char* fileC = (argc >= 7) ? argv[6] : nullptr;

    run_explicit(M, N, K, fileA, fileB, fileC);

    return 0;
}