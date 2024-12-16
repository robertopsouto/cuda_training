#include <stdio.h>

// Error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// Matrix side dimension
const size_t DSIZE = 16384;      
// CUDA maximum is 1024
const int block_size = 256;  

// Matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds)
{
  // Create typical 1D thread index from built-in variables
  int idx = FIXME 
  if (idx < ds)
  {
    float sum = 0.0f;
    // Write a for loop that will cause the thread
    // to iterate across a row, keeeping a running sum,
    // and write the result to sums
    for (size_t i = 0; i < ds; i++)
      sum += A[FIXME]
    sums[idx] = sum;
  }
}

// Matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds){
  // create typical 1D thread index from built-in variables
  int idx = FIXME 
  if (idx < ds)
  {
    float sum = 0.0f;
    // Write a for loop that will cause the thread 
    // to iterate down a column, keeeping a running sum, 
    // and write the result to sums
    for (size_t i = 0; i < ds; i++)
      sum += A[FIXME]
    sums[idx] = sum;
    }
}

bool validate(float *data, size_t sz)
{
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz)
      printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz); return false;
    return true;
}

int main()
{
  float *h_A, *h_sums, *d_A, *d_sums;
  // Allocate space for data in host memory
  h_A = new float[DSIZE*DSIZE];
  h_sums = new float[DSIZE]();
  
  // Initialize matrix in host memory
  for (int i = 0; i < DSIZE*DSIZE; i++)
    h_A[i] = 1.0f;
  
  // Allocate device space for A
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  
  FIXME // Allocate device space for vector d_sums
  
  cudaCheckErrors("cudaMalloc failure"); // error checking
  
  // Copy matrix A to device
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  
  // CUDA processing sequence step 1 is complete
  row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  
  // CUDA processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  
  // CUDA processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  
  printf("row sums correct!\n");
  
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  
  column_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  
  // CUDA processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  
  //CUDA processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("column sums correct!\n");
  
  return 0;
}
