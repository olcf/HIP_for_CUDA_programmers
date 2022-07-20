GPUCOMP  = hipcc  
BLAS = hipblas

INCLUDES  = -I$(OLCF_ESSL_ROOT)/include -I$(OLCF_HIP_CUDA_ROOT)/include
LIBRARIES = -L$(OLCF_ESSL_ROOT)/lib64 -lessl -L$(OLCF_HIP_CUDA_ROOT)/lib -l$(BLAS)

cpu_gpu_dgemm_hip: cpu_gpu_dgemm_hip.o
	$(GPUCOMP) $(LIBRARIES) cpu_gpu_dgemm_hip.o -o cpu_gpu_dgemm_hip

cpu_gpu_dgemm_hip.o: cpu_gpu_dgemm_hip.cpp
	$(GPUCOMP) $(INCLUDES) -c cpu_gpu_dgemm_hip.cpp
#	$(GPUCOMP) -w $(INCLUDES) -c cpu_gpu_dgemm_hip.cpp

.PHONY: clean

clean:
	rm -f cpu_gpu_dgemm_hip *.o
