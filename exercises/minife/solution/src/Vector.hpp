#ifndef _Vector_hpp_
#define _Vector_hpp_

//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

#include <vector>
#include <assert.h>
#include <thrust/device_vector.h>
#include <hip/hip_runtime.h>

namespace miniFE {

template<typename Scalar,
         typename LocalOrdinal,
         typename GlobalOrdinal>
struct PODVector {
  typedef Scalar ScalarType;
  typedef LocalOrdinal LocalOrdinalType;
  typedef GlobalOrdinal GlobalOrdinalType;
  
  Scalar *coefs;
  LocalOrdinal n;
  GlobalOrdinal startIndex;
};

template<typename Scalar,
         typename LocalOrdinal,
         typename GlobalOrdinal>
struct Vector {
  typedef Scalar ScalarType;
  typedef LocalOrdinal LocalOrdinalType;
  typedef GlobalOrdinal GlobalOrdinalType;

  Vector(GlobalOrdinal startIdx, LocalOrdinal local_sz)
   : startIndex(startIdx),
     local_size(local_sz),
     coefs(local_size),
     d_coefs(local_size,0)
  {
  }

  ~Vector()
  {
  }

  void copyToDevice(LocalOrdinal startIndex=0) const {
    int size=local_size-startIndex;
    hipMemcpy(const_cast<Scalar*>(thrust::raw_pointer_cast(&d_coefs[startIndex])),const_cast<Scalar*>(&coefs[startIndex]),sizeof(ScalarType)*size,hipMemcpyHostToDevice);
  }
  void copyToHost(LocalOrdinal startIndex=0) const {
    int size=local_size-startIndex;
    hipMemcpy(const_cast<Scalar*>(&coefs[startIndex]),const_cast<Scalar*>(thrust::raw_pointer_cast(&d_coefs[startIndex])),sizeof(ScalarType)*size,hipMemcpyDeviceToHost);
  }
  void copyToDeviceAsync(LocalOrdinal startIndex=0,hipStream_t s=0) const {
    int size=local_size-startIndex;
    hipMemcpyAsync(const_cast<Scalar*>(thrust::raw_pointer_cast(&d_coefs[startIndex])),const_cast<Scalar*>(&coefs[startIndex]),sizeof(ScalarType)*size,hipMemcpyHostToDevice,s);
  }
  void copyToHostAsync(LocalOrdinal startIndex=0,hipStream_t s=0) const {
    int size=local_size-startIndex;
    hipMemcpyAsync(const_cast<Scalar*>(&coefs[startIndex]),const_cast<Scalar*>(thrust::raw_pointer_cast(&d_coefs[startIndex])),sizeof(ScalarType)*size,hipMemcpyDeviceToHost,s);
  }

  PODVector<Scalar,LocalOrdinal,GlobalOrdinal> getPOD() const {
    PODVector<Scalar,LocalOrdinal,GlobalOrdinal> ret;
    ret.coefs=const_cast<Scalar*>(thrust::raw_pointer_cast(&d_coefs[0]));
    ret.n=local_size;
    ret.startIndex=startIndex;
    return ret;
  }

  GlobalOrdinal startIndex;
  LocalOrdinal local_size;
  std::vector<Scalar> coefs;
  thrust::device_vector<Scalar> d_coefs;
};


}//namespace miniFE

#endif

