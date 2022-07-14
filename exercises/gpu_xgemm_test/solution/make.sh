#!/bin/bash

echo $USE_GPU
if [[ $(hostname -d) == "summit"* ]]; then
module -q load gcc cuda/11.4.0 essl hip-cuda
make -f Makefile.summit clean
make -f Makefile.summit 
elif [[ $(hostname -d) == "crusher"* ]]; then
module load rocm openblas/0.3.17-omp
make -f Makefile.crusher clean
make -f Makefile.crusher  
fi

