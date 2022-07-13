/*------------------------------------------------------------------------------------------------
This program will fill 2 NxN matrices with random numbers, compute a matrix multiply on the CPU 
and then on the GPU, compare the values for correctness, and print _SUCCESS_ (if successful).

Written by Tom Papatheodore
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <essl.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <complex.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
        exit(0);                                                                            \
    }                                                                                     \
}while(0)

#define N 512

typedef std::complex<double> complexd ;

int main(int argc, char *argv[])
{

    // Set device to GPU 0
    cudaErrorCheck( cudaSetDevice(0) );

    /* Allocate memory for A, B, C on CPU ----------------------------------------------*/
    complexd *A = (complexd*)malloc(N*N*sizeof(complexd));
    complexd *B = (complexd*)malloc(N*N*sizeof(complexd));
    complexd *C = (complexd*)malloc(N*N*sizeof(complexd));

    /* Set Values for A, B, C on CPU ---------------------------------------------------*/

    // Max size of random double
    double max_value = 10.0;

    // Set A, B, C
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            A[i*N + j] = complexd((double)rand()/(double)(RAND_MAX/max_value) , (double)rand()/(double)(RAND_MAX/max_value));
            B[i*N + j] = complexd((double)rand()/(double)(RAND_MAX/max_value) , (double)rand()/(double)(RAND_MAX/max_value));
            C[i*N + j] = complexd(0.0 , 0.0);
        }
    }

    /* Allocate memory for d_A, d_B, d_C on GPU ----------------------------------------*/
    cuDoubleComplex *d_A, *d_B, *d_C;
    cudaErrorCheck( cudaMalloc(&d_A, N*N*sizeof(cuDoubleComplex)) );
    cudaErrorCheck( cudaMalloc(&d_B, N*N*sizeof(cuDoubleComplex)) );
    cudaErrorCheck( cudaMalloc(&d_C, N*N*sizeof(cuDoubleComplex)) );

    /* Copy host arrays (A,B,C) to device arrays (d_A,d_B,d_C) -------------------------*/
    cudaErrorCheck(cudaMemcpy(d_A, A, N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    cudaErrorCheck(cudaMemcpy(d_B, B, N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    cudaErrorCheck(cudaMemcpy(d_C, C, N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );	

    /* Perform Matrix Multiply on CPU --------------------------------------------------*/

    const complexd alpha = complexd(1.0, 1.0);
    const complexd beta = complexd(0.0, 0.0);

    const cuDoubleComplex cualpha = make_cuDoubleComplex(1.0, 1.0);
    const cuDoubleComplex cubeta = make_cuDoubleComplex(0.0, 0.0);

    zgemm("n", "n", N, N, N, alpha, A, N, B, N, beta, C, N);

    /* Perform Matrix Multiply on GPU --------------------------------------------------*/

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status = cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &cualpha, d_A, N, d_B, N, &cubeta, d_C, N);
    if (status != CUBLAS_STATUS_SUCCESS){
        printf("cublasZgemm failed with code %d\n", status);
        return EXIT_FAILURE;
    }

	/* Copy values of d_C back from GPU and compare with values calculated on CPU ------*/

    // Copy values of d_C (computed on GPU) into host array C_fromGPU	
    complexd *C_fromGPU = (complexd*)malloc(N*N*sizeof(complexd));	
    cudaErrorCheck( cudaMemcpy(C_fromGPU, d_C, N*N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    // Check if CPU and GPU give same results
    complexd tolerance = complexd(1.0e-13, 1.0e-13);
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(abs((C[i*N + j] - C_fromGPU[i*N + j])/C[i*N + j]) > abs(tolerance)){
		
                printf("Element C[%d][%d] (%f+i%f) and C_fromGPU[%d][%d] (%f+i%f) do not match!\n", i, j, real(C[i*N + j]), imag(C[i*N + j]), i, j, real(C_fromGPU[i*N + j]), imag(C_fromGPU[i*N + j]));
                return EXIT_FAILURE;
            }
    //        printf("Element C[%d][%d] (%f+i%f) and C_fromGPU[%d][%d] (%f+i%f)\n", i, j, real(C[i*N + j]), imag(C[i*N + j]), i, j, real(C_fromGPU[i*N + j]), imag(C_fromGPU[i*N + j]));
        }
    }

    /* Clean up and output --------------------------------------------------------------*/

    cublasDestroy(handle);

    // Free GPU memory
    cudaErrorCheck( cudaFree(d_A) );
    cudaErrorCheck( cudaFree(d_B) );
    cudaErrorCheck( cudaFree(d_C) );

    // Free CPU memory
    free(A);
    free(B);
    free(C);
    free(C_fromGPU);

    printf("__SUCCESS__\n");

    return 0;
}
