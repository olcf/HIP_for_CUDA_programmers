#include <hipblas.h>
#include <stdio.h>
#include "mm_gpu.h"

// Macro for checking errors in CUDA API calls
#define gpuErrorCheck(call)                                                              \
do{                                                                                       \
    hipError_t gpuErr = call;                                                             \
    if(hipSuccess != gpuErr){                                                             \
      printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)


int gpu_init(int node_local_rank )
{ 

	// Find how many GPUs CUDA runtime says are available
	int num_devices = 0;
	gpuErrorCheck( hipGetDeviceCount(&num_devices) );

	// Map MPI ranks to GPUs according to node-local MPI rank (round-robin)
	int gpu_id; 
	gpu_id = node_local_rank % num_devices;
	gpuErrorCheck( hipSetDevice(gpu_id) );
	return gpu_id;
}

float mm_on_gpu_and_verify(int loop_count, int N, double* A, double* B, double* C, double alpha, double beta ) {
    // Allocate memory for d_A, d_B, d_C on GPU
    double *d_A, *d_B, *d_C;
    gpuErrorCheck( hipMalloc(&d_A, N*N*sizeof(double)) );
    gpuErrorCheck( hipMalloc(&d_B, N*N*sizeof(double)) );
    gpuErrorCheck( hipMalloc(&d_C, N*N*sizeof(double)) );

	// Copy host arrays (A,B,C) to device arrays (d_A,d_B,d_C)
	gpuErrorCheck( hipMemcpy(d_A, A, N*N*sizeof(double), hipMemcpyHostToDevice) );
	gpuErrorCheck( hipMemcpy(d_B, B, N*N*sizeof(double), hipMemcpyHostToDevice) );
	gpuErrorCheck( hipMemcpy(d_C, C, N*N*sizeof(double), hipMemcpyHostToDevice) );	

	hipblasHandle_t handle;
	hipblasCreate(&handle);

	hipEvent_t start_gpu, end_gpu;
	hipEventCreate(&start_gpu);
	hipEventCreate(&end_gpu);

	// Start GPU timer
	hipEventRecord(start_gpu);

	for(int i=0; i<loop_count; i++){
		// Perform Matrix Multiply on GPU
		hipblasStatus_t status = hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
		if (status != HIPBLAS_STATUS_SUCCESS){
			printf("hipblasDgemm failed with code %d\n", status);
			return -1;
		}
	}

	// Stop GPU timer
	hipEventRecord(end_gpu);
	hipEventSynchronize(end_gpu);
	float milliseconds = 0.0;
	float seconds;

	hipEventElapsedTime(&milliseconds, start_gpu, end_gpu);
	seconds = milliseconds / 1000;

	hipblasDestroy(handle);

	// Copy values of d_C computed on GPU into host array C_fromGPU	
	double *C_fromGPU = (double*)malloc(N*N*sizeof(double));	
	gpuErrorCheck( hipMemcpy(C_fromGPU, d_C, N*N*sizeof(double), hipMemcpyDeviceToHost) );
	// Check if CPU and GPU give same results
	double tolerance = 1.0e-13;
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			if(fabs((C[i*N + j] - C_fromGPU[i*N + j])/C[i*N + j]) > tolerance){
				printf("Element C[%d][%d] (%f) and C_fromGPU[%d][%d] (%f) do not match!\n", i, j, C[i*N + j], i, j, C_fromGPU[i*N + j]);
				return -1;
			}
		}
	}

	// Free GPU memory
	gpuErrorCheck( hipFree(d_A) );
	gpuErrorCheck( hipFree(d_B) );
	gpuErrorCheck( hipFree(d_C) );

	free(C_fromGPU);
	return seconds;

}
