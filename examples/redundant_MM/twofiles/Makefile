CUCOMP  = nvcc
#CUCOMP  = nvcc -ccbin xlc++_r
MPICOMP = mpicxx
CUFLAGS = -arch=sm_70

INCLUDES  = -I${OLCF_ESSL_ROOT}/include
LIBRARIES = -L${OLCF_ESSL_ROOT}/lib64 -L${OLCF_CUDA_ROOT}/lib64 -lessl -lcublas -lcudart

redundant_mm: redundant_MM.o redundant_MM_gpu.o
	${MPICOMP} ${LIBRARIES} redundant_MM.o redundant_MM_gpu.o -o redundant_mm 

redundant_MM_gpu.o: redundant_MM_gpu.cu
	${CUCOMP} ${CUFLAGS}  -o redundant_MM_gpu.o -c redundant_MM_gpu.cu

redundant_MM.o: redundant_MM.cpp
	${MPICOMP} ${INCLUDES} -o redundant_MM.o -c redundant_MM.cpp

.PHONY: clean cleanall

clean:
	rm -f redundant_mm *.o

cleanall:
	rm -f redundant_mm *.o mat_mul* *.nvvp
