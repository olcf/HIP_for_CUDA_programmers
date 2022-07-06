#!/bin/bash

echo $USE_GPU
module -q load gcc cuda/11.4.0 essl

make -f Makefile.summit distclean
make -f Makefile.summit USE_GPU=${USE_GPU:-YES}

