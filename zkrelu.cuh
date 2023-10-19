#ifndef ZKRELU_CUH
#define ZKRELU_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "proof.cuh"

class zkReLU {
protected:
    FrTensor* sign_ptr = nullptr;
    FrTensor* mag_bin_ptr = nullptr;
    FrTensor* rem_bin_ptr = nullptr;
    void reset_ptrs(uint size);
public:
    FrTensor operator()(const FrTensor& X);
    void prove(const FrTensor& X, const FrTensor& Z);
    ~zkReLU();
};

DEVICE Fr_t ulong_to_scalar(unsigned long num);

DEVICE unsigned long scalar_to_ulong(Fr_t num);

KERNEL void relu_kernel(Fr_t* X, Fr_t* Z, Fr_t* sign, Fr_t* mag_bin, Fr_t* rem_bin, uint n);

#endif  // ZKRELU_CUH
