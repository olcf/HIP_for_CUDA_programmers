#include "hip/hip_runtime.h"
#include <stdio.h>


// Size of array
#define N 1048576

// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
	// Number of bytes to allocate for N doubles
	size_t bytes = N*sizeof(double);

	// Allocate memory for arrays A, B, and C on host
	double *A = (double*)malloc(bytes);
	double *B = (double*)malloc(bytes);
	double *C = (double*)malloc(bytes);

	// Allocate memory for arrays d_A, d_B, and d_C on device
	double *d_A, *d_B, *d_C;
	hipMalloc(&d_A, bytes);	
	hipMalloc(&d_B, bytes);
	hipMalloc(&d_C, bytes);

	// Fill host arrays A, B, and C
	for(int i=0; i<N; i++)
	{
		A[i] = 1.0;
		B[i] = 2.0;
		C[i] = 0.0;
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice);
	hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice);

	// Set execution configuration parameters
	//		thr_per_blk: number of CUDA threads per block
	//		blk_in_grid: number of blocks in grid
	int thr_per_blk = 256;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

	// Launch kernel
	hipLaunchKernelGGL(add_vectors, blk_in_grid, thr_per_blk , 0, 0, d_A, d_B, d_C);

  	// Check for errors in kernel launch (e.g. invalid execution configuration paramters)
	hipError_t deviceErrSync  = hipGetLastError();

  	// Check for errors on the GPU after control is returned to CPU
	hipError_t deviceErrAsync = hipDeviceSynchronize();

	if (deviceErrSync != hipSuccess) { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(deviceErrSync)); exit(0); }
	if (deviceErrAsync != hipSuccess) { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(deviceErrAsync)); exit(0); }

	// Copy data from device array d_C to host array C
	hipMemcpy(C, d_C, bytes, hipMemcpyDeviceToHost);

	// Verify results
    double tolerance = 1.0e-14;
	for(int i=0; i<N; i++)
	{
            if( fabs(C[i] - 3.0) > tolerance )
		{ 
			printf("Error: value of C[%d] = %f instead of 3.0\n", i, C[i]);
			exit(-1);
		}
	}	

	// Free CPU memory
	free(A);
	free(B);
	free(C);

	// Free GPU memory
	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);

	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", thr_per_blk);
	printf("Blocks In Grid    = %d\n", blk_in_grid);
	printf("---------------------------\n\n");

	return 0;
}
