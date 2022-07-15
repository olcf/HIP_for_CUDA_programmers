/*------------------------------------------------------------------------------------------------
This program will fill 2 NxN matrices with random numbers, compute a matrix multiply on the CPU 
and then on the GPU, compare the values for correctness, and print _SUCCESS_ (if successful).

Written by Tom Papatheodore
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <essl.h>
#include <hipblas.h>
#include <cublas_v2.h>
#include <hip/hip_complex.h>
#include <complex.h>

// Macro for checking errors in CUDA API calls
#define gpuErrorCheck(call)                                                              \
do{                                                                                       \
    hipError_t gpuErr = call;                                                             \
    if(hipSuccess != gpuErr){                                                             \
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr));\
        exit(0);                                                                            \
    }                                                                                     \
}while(0)

#define N 512

typedef std::complex<double> complexd ;

int main(int argc, char *argv[])
{

    // Set device to GPU 0
    gpuErrorCheck( hipSetDevice(0) );

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
    hipDoubleComplex *d_A, *d_B, *d_C;
    gpuErrorCheck( hipMalloc(&d_A, N*N*sizeof(hipDoubleComplex)) );
    gpuErrorCheck( hipMalloc(&d_B, N*N*sizeof(hipDoubleComplex)) );
    gpuErrorCheck( hipMalloc(&d_C, N*N*sizeof(hipDoubleComplex)) );

    /* Copy host arrays (A,B,C) to device arrays (d_A,d_B,d_C) -------------------------*/
    gpuErrorCheck(hipMemcpy(d_A, A, N*N*sizeof(hipDoubleComplex), hipMemcpyHostToDevice) );
    gpuErrorCheck(hipMemcpy(d_B, B, N*N*sizeof(hipDoubleComplex), hipMemcpyHostToDevice) );
    gpuErrorCheck(hipMemcpy(d_C, C, N*N*sizeof(hipDoubleComplex), hipMemcpyHostToDevice) );	

    /* Perform Matrix Multiply on CPU --------------------------------------------------*/

    const complexd alpha = complexd(1.0, 1.0);
    const complexd beta = complexd(0.0, 0.0);

    const hipDoubleComplex gpualpha = make_hipDoubleComplex(1.0, 1.0);
    const hipDoubleComplex gpubeta = make_hipDoubleComplex(0.0, 0.0);

    zgemm("n", "n", N, N, N, alpha, A, N, B, N, beta, C, N);

    /* Perform Matrix Multiply on GPU --------------------------------------------------*/

#ifdef USE_CUDA
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status = cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &gpualpha, d_A, N, d_B, N, &gpubeta, d_C, N);
    if (status != CUBLAS_STATUS_SUCCESS){
        printf("cublasZgemm3m failed with code %d\n", status);
        return EXIT_FAILURE;
    }
#else
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    // Replacing cublasZgemm3m with hipblasZgemm as HIP doesn't have an implementation of cublasZgemm3m
    hipblasStatus_t status = hipblasZgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, 
	(hipblasDoubleComplex*)&gpualpha,
	(hipblasDoubleComplex*)d_A,
	N, 
	(hipblasDoubleComplex*)d_B, 
	N, 
	(hipblasDoubleComplex*)&gpubeta, 
	(hipblasDoubleComplex*)d_C, 
	N);
    if (status != HIPBLAS_STATUS_SUCCESS){
        printf("hipblasZgemm failed with code %d\n", status);
        return EXIT_FAILURE;
    }
#endif

    /* Copy values of d_C back from GPU and compare with values calculated on CPU ------*/

    // Copy values of d_C (computed on GPU) into host array C_fromGPU	
    complexd *C_fromGPU = (complexd*)malloc(N*N*sizeof(complexd));	
    gpuErrorCheck( hipMemcpy(C_fromGPU, d_C, N*N*sizeof(hipDoubleComplex), hipMemcpyDeviceToHost) );

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

    hipblasDestroy(handle);

    // Free GPU memory
    gpuErrorCheck( hipFree(d_A) );
    gpuErrorCheck( hipFree(d_B) );
    gpuErrorCheck( hipFree(d_C) );

    // Free CPU memory
    free(A);
    free(B);
    free(C);
    free(C_fromGPU);

    printf("__SUCCESS__\n");

    return 0;
}
