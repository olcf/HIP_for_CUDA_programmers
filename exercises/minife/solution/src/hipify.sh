#!/bin/bash

hipify-perl CudaELLMatrix.hpp > tmp && mv tmp  GpuELLMatrix.hpp && rm CudaELLMatrix.hpp;
hipify-perl CudaHex8.hpp > tmp && mv tmp  GpuHex8.hpp && rm CudaHex8.hpp;
hipify-perl CudaUtils.cu > GpuUtils.cpp && rm CudaUtils.cu ;
