CUCOMP  = hipcc
BLAS = hipblas

INCLUDES  = -I$(OLCF_ESSL_ROOT)/include -I$(OLCF_HIP_CUDA_ROOT)/include
LIBRARIES = -L$(OLCF_HIP_CUDA_ROOT)/lib -L$(OLCF_ESSL_ROOT)/lib64 -l$(BLAS) -lessl

cpu_gpu_zgemm_hip: cpu_gpu_zgemm_hip.o
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) cpu_gpu_zgemm_hip.o -o cpu_gpu_zgemm_hip

cpu_gpu_zgemm_hip.o: cpu_gpu_zgemm_hip.cpp
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c cpu_gpu_zgemm_hip.cpp

.PHONY: clean

clean:
	rm -f cpu_gpu_zgemm_hip *.o
