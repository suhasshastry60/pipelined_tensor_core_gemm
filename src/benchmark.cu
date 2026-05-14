#include <cuda_fp16.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "gemm_kernel.h"

namespace {

bool getDims(const char* fname, int& rows, int& cols) {
    std::ifstream f(fname);
    if (!f.is_open()) return false;
    f >> rows >> cols;
    return true;
}

void readA(const char* fname, __half* mat) {
    std::ifstream f(fname);
    int rows, cols;
    float x;
    f >> rows >> cols;
    for (int i = 0; i < rows * cols; i++) {
        f >> x;
        mat[i] = __float2half(x);
    }
}

void readBColumnMajor(const char* fname, __half* mat) {
    std::ifstream f(fname);
    int rows, cols;
    float x;
    f >> rows >> cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            f >> x;
            mat[j * rows + i] = __float2half(x);
        }
    }
}

void readC(const char* fname, float* mat) {
    std::ifstream f(fname);
    int rows, cols;
    f >> rows >> cols;
    for (int i = 0; i < rows * cols; i++) {
        f >> mat[i];
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    bool usingFiles = false;

    if (argc == 4) {
        int parsedM = std::atoi(argv[1]);
        if (parsedM > 0) {
            M = parsedM;
            N = std::atoi(argv[2]);
            K = std::atoi(argv[3]);
        } else {
            usingFiles = true;
            int rowsA, colsA, rowsB, colsB, rowsC, colsC;
            if (!getDims(argv[1], rowsA, colsA) || !getDims(argv[2], rowsB, colsB) ||
                !getDims(argv[3], rowsC, colsC)) {
                std::cerr << "[ERROR] Could not open one or more input files\n";
                return 1;
            }
            M = rowsA;
            K = colsA;
            N = colsB;
        }
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [M N K | A.txt B.txt C.txt]\n";
        return 1;
    }

    __half* hA = static_cast<__half*>(std::malloc(M * K * sizeof(__half)));
    __half* hB = static_cast<__half*>(std::malloc(K * N * sizeof(__half)));
    float* hC = static_cast<float*>(std::malloc(M * N * sizeof(float)));
    float* hCRef = static_cast<float*>(std::malloc(M * N * sizeof(float)));

    if (!hA || !hB || !hC || !hCRef) {
        std::cerr << "[ERROR] Host allocation failed\n";
        return 1;
    }

    if (usingFiles) {
        readA(argv[1], hA);
        readBColumnMajor(argv[2], hB);
        readC(argv[3], hCRef);
    } else {
        for (int i = 0; i < M * K; i++) hA[i] = __float2half(0.1f);
        for (int i = 0; i < K * N; i++) hB[i] = __float2half(0.1f);
    }

    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;

    cudaMalloc(&dA, M * K * sizeof(__half));
    cudaMalloc(&dB, K * N * sizeof(__half));
    cudaMalloc(&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hA, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(float));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    launchGemmKernel(M, N, K, dA, dB, dC);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    const int iters = 10;
    for (int i = 0; i < iters; i++) {
        launchGemmKernel(M, N, K, dA, dB, dC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (usingFiles) {
        bool pass = true;
        for (int i = 0; i < M * N; i++) {
            if (std::abs(hC[i] - hCRef[i]) > 0.05f) {
                std::cout << "[FAIL] at " << i << " expected " << hCRef[i]
                          << " got " << hC[i] << "\n";
                pass = false;
                break;
            }
        }
        if (pass) std::cout << "[PASS] GPU output matches reference\n";
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avgMs = ms / iters;
    double tflops = (2.0 * M * N * K) / (avgMs * 1e9);

    std::cout << "Avg Execution Time : " << avgMs << " ms\n";
    std::cout << "Throughput         : " << tflops << " TFLOPS\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    std::free(hA);
    std::free(hB);
    std::free(hC);
    std::free(hCRef);

    return 0;
}
