int gpu_init(int node_local_rank);
float mm_on_gpu_and_verify(int loop_count, int N, double* A, double* B, double* C, double alpha, double beta );
