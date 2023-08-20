//#include "bls12-381.cuh"
#include "fr-tensor.cuh"
#include <iostream>

const uint NUM_THREAD = 1024;

KERNEL void Fr_elementwise_add(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr1[gid], arr2[gid]);
}

KERNEL void Fr_broadcast_add(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr[gid], x);
}

FrTensor FrTensor::operator+(const FrTensor & t) const
{
  if (size != t.size) throw std::runtime_error("Incompatible dimensions");
  FrTensor out(size);
  Fr_elementwise_add<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, t.gpu_data, out.gpu_data, size);
  return out;
}

FrTensor FrTensor::operator+(const Fr_t & x) const
{
  FrTensor out(size);
  Fr_broadcast_add<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, x, out.gpu_data, size);
  return out;
}

// FrTensor& operator+=(const FrTensor &);

// FrTensor& operator+=(const Fr_t &);