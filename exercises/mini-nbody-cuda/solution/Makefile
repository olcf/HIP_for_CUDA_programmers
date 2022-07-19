GPUCOMP  = hipcc -ccbin xlc++_r -Xcompiler -std=c++11

mini-nbody_hip: mini-nbody_hip.o
	${GPUCOMP} mini-nbody_hip.o -o mini-nbody_hip

mini-nbody_hip.o: mini-nbody_hip.cpp
	${GPUCOMP} -c mini-nbody_hip.cpp

.PHONY: clean

clean:
	rm -f mini-nbody_hip *.o
