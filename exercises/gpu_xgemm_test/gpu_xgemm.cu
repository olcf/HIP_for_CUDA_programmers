#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <essl.h>
#include <cublas_v2.h>
#include <getopt.h>

/* ---------------------------------------------------------------------------------
Macro for checking errors in CUDA API calls
----------------------------------------------------------------------------------*/
#define cudaErrorCheck(call)                                                                 \
do{                                                                                          \
    cudaError_t cudaErr = call;                                                              \
    if(cudaSuccess != cudaErr){                                                              \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cudaErr)); \
      exit(0);                                                                               \
    }                                                                                        \
}while(0)


/* ---------------------------------------------------------------------------------
Macro for checking success in cuBLAS API calls
----------------------------------------------------------------------------------*/
#define cublasCheck(call)                                       \
do{                                                             \
  cublasStatus_t cublas_stat = call;                            \
  if(CUBLAS_STATUS_SUCCESS != cublas_stat){                     \
    std::cout << "cublas call failed. Exiting..." << std::endl; \
    exit(1);                                                    \
  }                                                             \
}while(0)


/* ---------------------------------------------------------------------------------
Define default options
----------------------------------------------------------------------------------*/

// Size of arrays (default)
int N = 1024;

// Selector for sgemm (single) or dgemm (double) (default)
std::string precision = "double";


/* ---------------------------------------------------------------------------------
Parse command line arguments
----------------------------------------------------------------------------------*/
void print_help(){

    printf(
    "----------------------------------------------------------------\n"
    "Usage: ./gpu_xgemm [OPTIONS]\n\n"
    "--matrix_size=<value>, -m:       Size of matrices\n"
    "                                 (default is 1024)\n\n"
    " "
    "--precision=<value>,   -p:       <value> can be single or double\n"
    "                                 to select sgemm or dgemm\n"
    "                                 (default is double)\n\n"
    " "
    "--help,                -h:       Show help\n"
    "----------------------------------------------------------------\n"
    );
    exit(1);
}

void process_arguments(int argc, char *argv[]){

    const char* const short_options = "m:p:h";

    const option long_options[] = {
        {"matrix_size", optional_argument, nullptr, 'm'},
        {"precision",   optional_argument, nullptr, 'p'},
        {"help",        no_argument,       nullptr, 'h'},
        {nullptr,       no_argument,       nullptr,   0}
    };

    while(true){

        const auto opts = getopt_long(argc, argv, short_options, long_options, nullptr);

        if(-1 == opts){ break; }

        switch(opts){
            case 'm':
                N = std::stoi(optarg);
                break;
            case 'p':
                precision = std::string(optarg);
                break;
            case 'h':
            default:
                print_help();
                break;
        }
    }
}


/* ---------------------------------------------------------------------------------
Host xgemm wrappers - e.g., if you run with double precision, the double version
will be used.
----------------------------------------------------------------------------------*/
void host_xgemm(double alpha, double beta, double *a, double *b, double *c){

    std::cout << "\nRunning cblas_dgemm...\n";
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                     alpha, a, N, b, N, beta, c, N);
}

void host_xgemm(float alpha, float beta, float *a, float *b, float *c){

    std::cout << "\nRunning cblas_sgemm...\n";
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                     alpha, a, N, b, N, beta, c, N);
}


/* ---------------------------------------------------------------------------------
Device xgemm wrappers - e.g., if you run with double precision, the double version 
will be used.
----------------------------------------------------------------------------------*/
cublasStatus_t device_xgemm(cublasHandle_t handle, double alpha, double beta, double *d_a, double *d_b, double *d_c){

    std::cout << "\nRunning cublasDgemm...\n";
    return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                        &alpha, d_a, N, d_b, N, &beta, d_c, N);
}

cublasStatus_t device_xgemm(cublasHandle_t handle, float alpha, float beta, float *d_a, float *d_b, float *d_c){

    std::cout << "\nRunning cublasSgemm...\n";
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                        &alpha, d_a, N, d_b, N, &beta, d_c, N);
}


/* ---------------------------------------------------------------------------------
Templated device xgemm test - e.g., if you run with double precision,
the type will be double (so read T as double).
----------------------------------------------------------------------------------*/
template <typename T>
void xgemm_test(T machine_eps){

    // Set which device to use
    int dev_id = 0;
    cudaErrorCheck( cudaSetDevice(dev_id) );

    // Scaling factors
    T alpha = (T)1.0;
    T beta  = (T)0.0;

    // Size (in bytes) of individual arrays
    int buffer_size = N * N * sizeof(T);

    // Host matrix buffers
    T *A   = (T*)malloc(buffer_size);
    T *B   = (T*)malloc(buffer_size);
    T *C   = (T*)malloc(buffer_size);
    T *rhs = (T*)malloc(buffer_size);

    // Device matrix buffers
    T *d_A, *d_B, *d_C;
    cudaErrorCheck( cudaMalloc(&d_A, buffer_size) );
    cudaErrorCheck( cudaMalloc(&d_B, buffer_size) );
    cudaErrorCheck( cudaMalloc(&d_C, buffer_size) );

    // Host buffers for test matrices that will hold the
    // correct answer from the host.
    T *test_A = (T*)malloc(buffer_size);
    T *test_B = (T*)malloc(buffer_size);
    T *test_C = (T*)malloc(buffer_size);

    // Fill matrics with random values
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            int index = i * N + j;         
            A[index] = (T)rand()/(T)RAND_MAX;
            B[index] = (T)rand()/(T)RAND_MAX;
            C[index] = (T)0.0;
        }
    }

    // Copy same values to test matrices
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            int index = i * N + j;
            test_A[index] = A[index];
            test_B[index] = B[index];
            test_C[index] = C[index];
        }
    }

    // Quantification of roundoff error for dot product (Golub Van Loan Matrix Computations 4th Edition)
    //     |fl((x^T)y) - (x^T)y)| <= 1.01*n*u*(|x|^T)|y|
    //       -> |C[index] - test_C[index]| <= rhs[index] can be used for correctness test
    //
    //     where
    //       -> fl((x^T)y) is GPU-computed value of each matrix element (i.e., dot product) of C
    //       -> (x^T)y is CPU-computed value of each matrix element (acting as correct answer) of test_C
    //       -> n is vector length N (i.e., size of square matrix)
    //       -> u is machine epsilon machine_epsilon

    T one_point_zero_one = (T)1.01;
    
    // Create rhs matrix, which holds the individual dot-product rhs values as shown above
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            int index = i * N + j;
            rhs[index] = (T)0.0;

            for(int k=0; k<N; k++){

                int index_A = i * N + k;
                int index_B = k * N + j;
                rhs[index] = rhs[index] + one_point_zero_one * (T)N * (T)machine_eps * fabs(A[index_A]) * fabs(B[index_B]);
            }

        }
    }

    // Pass host buffers to device buffers
    cudaErrorCheck( cudaMemcpy(d_A, A, buffer_size, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_B, B, buffer_size, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_C, C, buffer_size, cudaMemcpyHostToDevice) );

    // Call host xgemm routine - e.g., if using double precision, the double-precision
    // version of the wrapper will be used.
    host_xgemm(alpha, beta, test_A, test_B, test_C);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCheck( cublasCreate(&handle) );

    // Call device_xgemm routine - e.g., if using double precision, the double-precision
    // version of the wrapper will be used.
    cublasCheck( device_xgemm(handle, alpha, beta, d_A, d_B ,d_C) );

    // Copy results from device to host
    cudaErrorCheck( cudaMemcpy(C, d_C, buffer_size, cudaMemcpyDeviceToHost) );

    // Make sure host and device found the same results
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            int index = i * N + j;
            T error = fabs(C[index] - test_C[index]);

            if(error > rhs[index]){
                std::cout << "\n!!!!!!!!!!!!!!!!!!!!"                   << std::endl;
                std::cout << "error = " << error << " > " << rhs[index] << std::endl;
                std::cout << "Exiting..."                               << std::endl;
                std::cout << "!!!!!!!!!!!!!!!!!!!!"                     << std::endl;
                exit(1);
            }
        }
    }    

    // Clean up contexts and memory
    cublasCheck( cublasDestroy(handle) );

    cudaErrorCheck( cudaFree(d_A) );
    cudaErrorCheck( cudaFree(d_B) );
    cudaErrorCheck( cudaFree(d_C) );

    free(A);
    free(B);
    free(C);
    free(test_A);
    free(test_B);
    free(test_C);
    free(rhs);
}


/* ---------------------------------------------------------------------------------
Main program
----------------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    process_arguments(argc, argv);

    std::cout << "\n-----------------------------\n" << std::endl;
    std::cout << "N = " << N                         << std::endl;
    std::cout << "precision = " << precision.c_str() << std::endl;

    if(precision == "double"){
        double machine_epsilon = (double)1.0e-16;
        std::cout << "machine epsilon = " << machine_epsilon << std::endl;
        std::cout << "\n-----------------------------"       << std::endl;       
        xgemm_test<double>(machine_epsilon);
    }
    else if(precision == "single"){
        float machine_epsilon = (float)3.0e-8;
        std::cout << "machine epsilon = " << machine_epsilon << std::endl;
        std::cout << "\n-----------------------------"       << std::endl;
        xgemm_test<float>(machine_epsilon);
    }
    else{
        std::cout << "Must choose either double or single for precision. Exiting..." << std::endl;
        exit(1);
    }

    std::cout << "\n__SUCCESS__\n" << std::endl;    

    return 0;
}
