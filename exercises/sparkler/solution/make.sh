#!/bin/bash

echo $USE_GPU
if [[ $(hostname -d) == "summit"* ]]; then
module -q load gcc cuda/11.5.2 essl hip-cuda
make -f Makefile.summit distclean
make -f Makefile.summit USE_GPU=${USE_GPU:-YES}
elif [[ $(hostname -d) == "crusher"* ]]; then
module load rocm openblas/0.3.17-omp
make -f Makefile.crusher distclean
make -f Makefile.crusher USE_GPU=${USE_GPU:-YES} 
fi

