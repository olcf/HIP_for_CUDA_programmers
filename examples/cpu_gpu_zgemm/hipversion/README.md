This program will fill 2 NxN matrices with random complex numbers, 
compute a matrix multiply on the CPU and then on the GPU, 
compare the values for correctness, and print SUCCESS (if successful).

Make sure you load the following modules

```
module load cuda/11.5.2 hip-cuda essl
```

Compile with `make` for the `hipblasZgemm` call. 
Compile with `make USE_CUDA=YES` for the `cublasZgemm3m` call.
`make USE_CUDA=yes` will use a mix of hip and cuda calls in the same
file.
