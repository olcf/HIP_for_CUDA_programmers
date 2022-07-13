CUCOMP  = nvcc
BLAS = cublas
CUFLAGS = -arch=sm_70

INCLUDES  = -I$(OLCF_ESSL_ROOT)/include
LIBRARIES = -L$(CUDA_DIR)/lib64 -L$(OLCF_ESSL_ROOT)/lib64 -l$(BLAS) -lessl

cpu_gpu_zgemm_cuda: cpu_gpu_zgemm.o
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) cpu_gpu_zgemm.o -o cpu_gpu_zgemm_cuda

cpu_gpu_zgemm.o: cpu_gpu_zgemm.cu
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c cpu_gpu_zgemm.cu

.PHONY: clean

clean:
	rm -f cpu_gpu_zgemm *.o
