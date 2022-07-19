#include <stdio.h>

// Macro for checking errors in GPU API calls
#define cudaErrorCheck(call)                                                                 \
do{                                                                                         \
    cudaError_t cudaErr = call;                                                               \
    if(cudaSuccess != cudaErr){                                                               \
        printf("GPU Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cudaErr)); \
        exit(1);                                                                            \
    }                                                                                       \
}while(0)

// Values for NxN matrix
#define N 4096

__global__ void matrix_multiply(double *a, double *b, double *c)
{
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row    = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < N && column < N)
    {
        // dot product of row, column
        double element = 0.0;
        for(int i=0; i<N; i++){
            element += a[row * N + i] * b[i * N + column];
        }
        
        c[row * N + column] = element;
    }
}

// Main program
int main()
{
    // Number of bytes to allocate for NxN matrix
    size_t bytes = N*N*sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double A[N][N];
    double B[N][N];
    double C[N][N];

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    cudaErrorCheck( cudaMalloc(&d_A, bytes) );
    cudaErrorCheck( cudaMalloc(&d_B, bytes) );
    cudaErrorCheck( cudaMalloc(&d_C, bytes) );

    // Initialize host arrays A, B, C
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            if(j % 2 == 0){
                A[i][j] = sin(j);
            }
            else{
                A[i][j] = cos(j-1);
            }

            if(i % 2 == 0){
                B[i][j] = sin(i);
            }
            else{
                B[i][j] = cos(i-1);
            }

            C[i][j] = 0.0;
        }
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) );

    // Set execution configuration parameters
    // 		threads_per_block: number of GPU threads per grid block
    //		blocks_in_grid   : number of blocks in grid
    //		(These are c structs with 3 member variables x, y, x)
    dim3 threads_per_block( 16, 16, 1 );
    dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(N) / threads_per_block.y ), 1 );

    // Launch kernel
    matrix_multiply<<<blocks_in_grid, threads_per_block>>>( d_A, d_B, d_C);

    // Check for synchronous errors during kernel launch (e.g. invalid execution configuration paramters)
    cudaErrorCheck( cudaGetLastError() );

    // Check for asynchronous errors during GPU execution (after control is returned to CPU)
    cudaErrorCheck( cudaDeviceSynchronize() );

    // Copy data from device array d_C to host array C
    cudaErrorCheck( cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost) );

    // Verify results
    double tolerance = 1.0e-12;
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            if( fabs(C[i][j] - (double)N/2.0) > tolerance)
            {
                printf("C[%d][%d] = %0.14f instead of (N/2) = %0.14f\n", i, j, C[i][j], (double)N/2.0);
                exit(1);
            }
        }
    }

    // Free GPU memory
    cudaErrorCheck( cudaFree(d_A) );
    cudaErrorCheck( cudaFree(d_B) );
    cudaErrorCheck( cudaFree(d_C) );

    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("N                         = %d\n", N);
    printf("Threads Per Block (x-dim) = %d\n", threads_per_block.x);
    printf("Threads Per Block (y-dim) = %d\n", threads_per_block.y);
    printf("Blocks In Grid (x-dim)    = %d\n", blocks_in_grid.x);
    printf("Blocks In Grid (y-dim)    = %d\n", blocks_in_grid.y);
    printf("--------------------------------\n\n");

    return 0;
}
