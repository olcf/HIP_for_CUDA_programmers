Make sure you load the following modules

```
module load cuda/11.5.2 hip-cuda
```

Compile with
```
# for cuda
nvcc -o execname filename.cu
# for hip
hipcc -o execname filename.cpp
```
