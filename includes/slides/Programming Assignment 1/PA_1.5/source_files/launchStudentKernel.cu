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

void launchStudentKernel(int M, int N, int K, __half* A,
                                              __half* B,
                                              float* C) {

  // Launch your kernel here with appropriate grid and block sizes...

}
