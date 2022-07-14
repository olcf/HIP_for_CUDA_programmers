GPUCC    = hipcc

INCLUDES  = -I${OLCF_HIP_CUDA_ROOT}/include -I${OLCF_ESSL_ROOT}/include
LIBRARIES = -L${OLCF_HIP_CUDA_ROOT}/lib -lhipblas -L${OLCF_ESSL_ROOT}/lib64 -lessl

gpu_xgemm_hip: gpu_xgemm_hip.o
	$(GPUCC) $(LIBRARIES) gpu_xgemm_hip.o -o gpu_xgemm_hip

gpu_xgemm_hip.o: gpu_xgemm_hip.cpp
	$(GPUCC) $(INCLUDES) -c gpu_xgemm_hip.cpp

.PHONY: clean

clean:
	rm -f gpu_xgemm_hip *.o
