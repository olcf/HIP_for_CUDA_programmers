GPUCOMP  = nvcc
GPUFLAGS = -arch=sm_70

mini-nbody: mini-nbody.o
	${GPUCOMP} ${GPUFLAGS} mini-nbody.o -o mini-nbody

mini-nbody.o: mini-nbody.cu
	${GPUCOMP} ${GPUFLAGS} -c mini-nbody.cu

.PHONY: clean

clean:
	rm -f mini-nbody *.o
