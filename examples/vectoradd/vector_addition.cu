#include <stdio.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)

// Size of array
#define N 1048576

// Kernel
__global__ void cudaAddVectors(double *a, double *b, double *c)
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
	cudaErrorCheck( cudaMalloc(&d_A, bytes) );	
	cudaErrorCheck( cudaMalloc(&d_B, bytes) );
	cudaErrorCheck( cudaMalloc(&d_C, bytes) );

	// Fill host arrays A, B, and C
	for(int i=0; i<N; i++)
	{
		A[i] = 1.0;
		B[i] = 2.0;
        C[i] = 0.0;
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	cudaErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) );

	// Set execution configuration parameters
	//		thr_per_blk: number of CUDA threads per grid block
	//		blk_in_grid: number of blocks in grid
	int thr_per_blk = 256;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

	// Launch kernel
	add_vectors_cuda<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);

  	// Check for errors in kernel launch (e.g. invalid execution configuration paramters)
	cudaError_t cudaErrSync  = cudaGetLastError();

  	// Check for errors on the GPU after control is returned to CPU
	cudaError_t cudaErrAsync = cudaDeviceSynchronize();

	if (cudaErrSync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cudaErrSync)); exit(0); }
	if (cudaErrAsync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cudaErrAsync)); exit(0); }

	// Copy data from device array d_C to host array C
	cudaErrorCheck( cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost) );

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
	cudaErrorCheck( cudaFree(d_A) );
	cudaErrorCheck( cudaFree(d_B) );
	cudaErrorCheck( cudaFree(d_C) );

	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", thr_per_blk);
	printf("Blocks In Grid    = %d\n", blk_in_grid);
	printf("---------------------------\n\n");

	return 0;
}
