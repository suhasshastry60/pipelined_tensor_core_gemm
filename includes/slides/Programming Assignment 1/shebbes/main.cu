// Standard libraries you might need. You can add/remove as you like.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
using namespace std;
// This is where you allocate GPU memory, launch kernels and measure timings. 
// You'll need this when writing your report.

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
                valA = A[row_idx * K + k];
            else
                valA = A[k * M + row_idx];     

            // B is K x N
            if (layoutB == 0) 
                valB = B[k * N + col_idx];     
            else              
                valB = B[col_idx * K + k];     

            partial_sum += valA * valB;
        }
        C[row_idx * N + col_idx] = partial_sum;
    }
}


void readMatrixFile(const char* filename, float* data, int size) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    //This skips the first 3 metadata info
    int layout, rows, cols;
    file >> layout >> rows >> cols;

    for (int i = 0; i < size; i++) {
        file >> data[i];
    }
    file.close();
}

//This initializes the matrix to random numbers
void initRandom(float* data, int size) {
    for(int i = 0; i < size; ++i) data[i] = static_cast<float>(rand()) / RAND_MAX;
}

//This checks the output from sample_A and sample_B with sample_C
void checkResult(float* calculated, const char* refFile, int size) {
    if (refFile == nullptr) return;

    vector<float> reference(size);
    readMatrixFile(refFile, reference.data(), size); 

    bool passed = true;
    for (int i = 0; i < size; i++) {
        if (fabs(calculated[i] - reference[i]) > 1e-2) {
            cout << "[FAIL] Mismatch at index " << i << ". Calc=" << calculated[i] 
                      << " Ref=" << reference[i] << endl;
            passed = false;
            break; 
        }
    }
    if (passed) cout << "[SUCCESS] Result matches " << refFile << endl;
}

// --- 1. EXPLICIT MEMORY MODEL ---
void run_explicit(int M, int N, int K, 
                  const char* fileA = nullptr, 
                  const char* fileB = nullptr, 
                  const char* fileC = nullptr) {
    
    cout << "\n--- Explicit Memory (cudaMalloc) ---" << endl;
    size_t bytesA = M*K*sizeof(float);
    size_t bytesB = K*N*sizeof(float);
    size_t bytesC = M*N*sizeof(float);

    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);

    if (fileA && fileB) {
        cout << "Reading inputs from files..." << endl;
        readMatrixFile(fileA, h_A, M*K);
        readMatrixFile(fileB, h_B, K*N);
    } else {
        cout << "Initializing random data..." << endl;
        initRandom(h_A, M*K);
        initRandom(h_B, K*N);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    // dim3 block(8, 8); // I was testing for 8,8 to check for performance variations
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul<<<grid, block>>>(M, N, K, 0, 0, d_A, d_B, d_C); 
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Time: " << ms << " ms" << endl;

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    
    if (fileC) 
        checkResult(h_C, fileC, M*N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// --- 2. UNIFIED MEMORY MODEL ---
//need to comment out entire function when we use gpgpu sim
void run_unified(int M, int N, int K, 
                 const char* fileA = nullptr, 
                 const char* fileB = nullptr, 
                 const char* fileC = nullptr) {
    
    cout << "\n--- Unified Memory (cudaMallocManaged) ---" << endl;
    size_t bytesA = M*K*sizeof(float);
    size_t bytesB = K*N*sizeof(float);
    size_t bytesC = M*N*sizeof(float);

    float *u_A, *u_B, *u_C;
    cudaMallocManaged(&u_A, bytesA);
    cudaMallocManaged(&u_B, bytesB);
    cudaMallocManaged(&u_C, bytesC);

    if (fileA && fileB) {
        cout << "Reading inputs from files..." << endl;
        readMatrixFile(fileA, u_A, M*K);
        readMatrixFile(fileB, u_B, K*N);
    } else {
        cout << "Initializing random data..." << endl;
        initRandom(u_A, M*K);
        initRandom(u_B, K*N);
    }

    dim3 block(16, 16);
    // dim3 block(8, 8);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul<<<grid, block>>>(M, N, K, 0, 0, u_A, u_B, u_C); 
    cudaEventRecord(stop);

    cudaDeviceSynchronize(); 
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Time: " << ms << " ms" << endl;

    if (fileC) checkResult(u_C, fileC, M*N);

    cudaFree(u_A); cudaFree(u_B); cudaFree(u_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./main <M> <N> <K> [fileA] [fileB] [fileC]" << endl;
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    const char* fileA = (argc >= 6) ? argv[4] : nullptr;
    const char* fileB = (argc >= 6) ? argv[5] : nullptr;
    const char* fileC = (argc >= 7) ? argv[6] : nullptr;

    run_explicit(M, N, K, fileA, fileB, fileC);
    run_unified(M, N, K, fileA, fileB, fileC); // need to comment out when we run it on gpgpu sim

    return 0;
}