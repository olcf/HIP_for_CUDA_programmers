#include <cublas_v2.h>
#include <stdio.h>
#include "mm_gpu.h"

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)


int gpu_init(int node_local_rank )
{ 

	// Find how many GPUs CUDA runtime says are available
	int num_devices = 0;
	cudaErrorCheck( cudaGetDeviceCount(&num_devices) );

	// Map MPI ranks to GPUs according to node-local MPI rank (round-robin)
	int gpu_id; 
	gpu_id = node_local_rank % num_devices;
	cudaErrorCheck( cudaSetDevice(gpu_id) );
	return gpu_id;
}

float mm_on_gpu_and_verify(int loop_count, int N, double* A, double* B, double* C, double alpha, double beta ) {
    // Allocate memory for d_A, d_B, d_C on GPU
    double *d_A, *d_B, *d_C;
    cudaErrorCheck( cudaMalloc(&d_A, N*N*sizeof(double)) );
    cudaErrorCheck( cudaMalloc(&d_B, N*N*sizeof(double)) );
    cudaErrorCheck( cudaMalloc(&d_C, N*N*sizeof(double)) );

	// Copy host arrays (A,B,C) to device arrays (d_A,d_B,d_C)
	cudaErrorCheck( cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_B, B, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_C, C, N*N*sizeof(double), cudaMemcpyHostToDevice) );	

	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaEvent_t start_gpu, end_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&end_gpu);

	// Start GPU timer
	cudaEventRecord(start_gpu);

	for(int i=0; i<loop_count; i++){
		// Perform Matrix Multiply on GPU
		cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
		if (status != CUBLAS_STATUS_SUCCESS){
			printf("cublasDgemm failed with code %d\n", status);
			return -1;
		}
	}

	// Stop GPU timer
	cudaEventRecord(end_gpu);
	cudaEventSynchronize(end_gpu);
	float milliseconds = 0.0;
	float seconds;

	cudaEventElapsedTime(&milliseconds, start_gpu, end_gpu);
	seconds = milliseconds / 1000;

	cublasDestroy(handle);

	// Copy values of d_C computed on GPU into host array C_fromGPU	
	double *C_fromGPU = (double*)malloc(N*N*sizeof(double));	
	cudaErrorCheck( cudaMemcpy(C_fromGPU, d_C, N*N*sizeof(double), cudaMemcpyDeviceToHost) );
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
	cudaErrorCheck( cudaFree(d_A) );
	cudaErrorCheck( cudaFree(d_B) );
	cudaErrorCheck( cudaFree(d_C) );

	free(C_fromGPU);
	return seconds;

}
