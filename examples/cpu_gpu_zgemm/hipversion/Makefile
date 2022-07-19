GPUCOMP  = hipcc

USE_CUDA = NO

ifeq ($(USE_CUDA), YES)
INCLUDES  = -I$(OLCF_ESSL_ROOT)/include -I$(OLCF_HIP_CUDA_ROOT)/include -DUSE_CUDA
LIBRARIES = -L$(OLCF_HIP_CUDA_ROOT)/lib -lhipblas -L$(OLCF_ESSL_ROOT)/lib64  -lessl -L$(CUDA_DIR)/lib64 -lcublas
else
INCLUDES  = -I$(OLCF_ESSL_ROOT)/include -I$(OLCF_HIP_CUDA_ROOT)/include
LIBRARIES = -L$(OLCF_HIP_CUDA_ROOT)/lib -lhipblas  -L$(OLCF_ESSL_ROOT)/lib64  -lessl 
endif


cpu_gpu_zgemm_hip: cpu_gpu_zgemm_hip.o
	$(GPUCOMP) $(CUFLAGS) $(LIBRARIES) cpu_gpu_zgemm_hip.o -o cpu_gpu_zgemm_hip

cpu_gpu_zgemm_hip.o: cpu_gpu_zgemm_hip.cpp
	$(GPUCOMP) $(CUFLAGS) $(INCLUDES) -c cpu_gpu_zgemm_hip.cpp

.PHONY: clean

clean:
	rm -f cpu_gpu_zgemm_hip *.o
