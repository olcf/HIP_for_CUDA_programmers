GPUCC    = nvcc
GPUFLAGS = -arch=sm_70

INCLUDES  = -I${OLCF_CUDA_ROOT}/include -I${OLCF_ESSL_ROOT}/include
LIBRARIES = -L${OLCF_CUDA_ROOT}/lib64 -lcublas -L${OLCF_ESSL_ROOT}/lib64 -lessl

gpu_xgemm: gpu_xgemm.o
	$(GPUCC) $(GPUFLAGS) $(LIBRARIES) gpu_xgemm.o -o gpu_xgemm

gpu_xgemm.o: gpu_xgemm.cu
	$(GPUCC) $(GPUFLAGS) $(INCLUDES) -c gpu_xgemm.cu

.PHONY: clean

clean:
	rm -f gpu_xgemm *.o
