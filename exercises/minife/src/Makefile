#-----------------------------------------------------------------------

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -O3
CXXFLAGS = -O3

CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK -DMATVEC_OVERLAP -DMINIFE_HAVE_CUDA 

LDFLAGS= 
LIBS= -l nvToolsExt

#openmpi
MPICFLAGS=-I$(OMPI_DIR)/include -pthread 
#MPILDFLAGS=-L$(OMPI_DIR)/lib -lmpi_ibm -lmpi_cxx -ldl -lm -lrt -lnsl -lutil -lm -ldl
MPILDFLAGS=-L$(OMPI_DIR)/lib -lmpi_ibm 

NVCCFLAGS=-lineinfo -gencode=arch=compute_70,code=\"sm_70,compute_70\" 

NVCC=nvcc -Xcompiler -fopenmp 
CXX=mpicxx
CC=mpicc

include make_targets
