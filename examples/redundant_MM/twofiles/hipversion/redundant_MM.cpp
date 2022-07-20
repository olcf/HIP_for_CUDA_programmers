/*------------------------------------------------------------------------------------------------
redundant_MM

For each MPI rank, this program does the following:
  * Fill 2 NxN matrices with random numbers
  * Compute a matrix multiply on the CPU
	* Compute a matrix multiply on the GPU (loop_count times)
  * Compare the CPU and GPU results for consistency
  * Output the total runtime and time spent computing on the GPUs for each rank (and max)
    as well as the hardware thread and GPU used on a specific node

USAGE:

Two command line arguments must be supplied:
	N (matrix size)
	loop_count (number of times cublasDgemm is called)

For example,

	$ jsrun -n6 -c1 -g1 -a1 -r3 ./redundant_MM 2048 1000 | sort
	(N = 2048) Max Total Time: 6.879220 Max GPU Time: 2.816899
	Rank 000, HWThread 002, GPU 0, Node h41n09 - Total Time: 6.855115 GPU Time: 2.804994
	Rank 001, HWThread 004, GPU 1, Node h41n09 - Total Time: 6.816647 GPU Time: 2.814934
	Rank 002, HWThread 008, GPU 2, Node h41n09 - Total Time: 6.879220 GPU Time: 2.816899
	Rank 003, HWThread 000, GPU 0, Node h41n10 - Total Time: 5.862273 GPU Time: 2.814339
	Rank 004, HWThread 005, GPU 1, Node h41n10 - Total Time: 5.798143 GPU Time: 2.765094
	Rank 005, HWThread 010, GPU 2, Node h41n10 - Total Time: 5.746687 GPU Time: 2.785626

Written by Tom Papatheodore
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sched.h>
#include <mpi.h>
#include <essl.h>
#include "mm_gpu.h"

int main(int argc, char *argv[])
{

	/* -------------------------------------------------------------------------------------------
		MPI Initialization 
	--------------------------------------------------------------------------------------------*/
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char name[MPI_MAX_PROCESSOR_NAME];
	int resultlength;
	MPI_Get_processor_name(name, &resultlength);
	
	const char* nl_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
	int node_local_rank = atoi(nl_rank);

	int gpu_id;
	gpu_id = gpu_init(node_local_rank);

	/* -------------------------------------------------------------------------------------------
		Other Initialization 
	--------------------------------------------------------------------------------------------*/

	// Start Total Runtime Timer
	double start_time, end_time, elapsed_time;
	start_time = MPI_Wtime();

	// Matrix size
	int N;

	// Number of times cublasDgemm is called
	int loop_count;

	// Check for proper command line arguments
	if(argc != 3){
		printf("Must supply two arguments: N (matrix size) and loop_count (number of cublasDgemm calls). Exiting...\n");
		exit(0);
	}
	else{
		for(int i=0; i<strlen(argv[1]); i++){
			if(!isdigit(argv[1][i])){
				printf("1st argument must be a positive integer! Exiting...\n");
				exit(0);
			}
		}
		N = atoi(argv[1]);		

		for(int i=0; i<strlen(argv[2]); i++){
			if(!isdigit(argv[2][i])){
				printf("2nd argument must be a positive integer! Exiting...\n");
				exit(0);
			}
		}
		loop_count = atoi(argv[2]);
	}

	// Find hardware thread being used by each MPI rank
	int hwthread = sched_getcpu();


	/* -------------------------------------------------------------------------------------------
		Allocate memory for arrays on CPU and GPU
	--------------------------------------------------------------------------------------------*/

	// Allocate memory for A, B, C on CPU
	double *A = (double*)malloc(N*N*sizeof(double));
	double *B = (double*)malloc(N*N*sizeof(double));
	double *C = (double*)malloc(N*N*sizeof(double));


    /* -------------------------------------------------------------------------------------------
        Fill arrays on CPU
    --------------------------------------------------------------------------------------------*/

	// Max size of random double
	double max_value = 10.0;

	// Set A, B, and C
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			A[i*N + j] = (double)rand()/(double)(RAND_MAX/max_value);
			B[i*N + j] = (double)rand()/(double)(RAND_MAX/max_value);
			C[i*N + j] = 0.0;
		}
	}


	/* -------------------------------------------------------------------------------------------
		Perform DGEMM on CPU
	--------------------------------------------------------------------------------------------*/

	const double alpha = 1.0;
	const double beta = 0.0;

	// Perform Matrix Multiply on CPU
	dgemm("n", "n", N, N, N, alpha, A, N, B, N, beta, C, N);

    /* -------------------------------------------------------------------------------------------
        Perform DGEMM on GPU (loop_count times) and time GPU execution
    --------------------------------------------------------------------------------------------*/

	float seconds = mm_on_gpu_and_verify(loop_count, N, A, B, C, alpha, beta);
	if(seconds == -1) 
		return EXIT_FAILURE;




	// Free CPU memory
	free(A);
	free(B);
	free(C);

	// End Total Runtime Timer
	end_time = MPI_Wtime();
	elapsed_time = end_time - start_time;

	/* -------------------------------------------------------------------------------------------
		MPI Reductions to find the maximum total runtime and maximum time spent computing on GPUs.
		(These are used as proxies for total runtime and total time spent computing on GPUs)
	--------------------------------------------------------------------------------------------*/

	double total_time_max;
	MPI_Reduce(&elapsed_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	float gpu_time_max;
	MPI_Reduce(&seconds, &gpu_time_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

	/* -------------------------------------------------------------------------------------------
		Output and finalize
	--------------------------------------------------------------------------------------------*/

	// MPI rank 0 will output the maximum total runtime and maximum time spent computing on GPUs
	if(rank == 0){
		printf("(N = %d) Max Total Time: %f Max GPU Time: %f\n", N, total_time_max, gpu_time_max);
	}

	// Each MPI rank will output its total runtime and time spent computing on GPUs
	printf("Rank %03d, HWThread %03d, GPU %d, Node %s - Total Time: %f GPU Time: %f\n", rank, hwthread, gpu_id, name, elapsed_time, seconds); 

	MPI_Finalize();

	return 0;
}
