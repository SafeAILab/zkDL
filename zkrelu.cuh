#ifndef ZKRELU_CUH
#define ZKRELU_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 

class zkReLU {
protected:
    FrTensor* sign_ptr;
    FrTensor* mag_bin_ptr;
    FrTensor* rem_bin_ptr;
    void reset_ptrs(uint size);
public:
    FrTensor operator()(const FrTensor& X);
    ~zkReLU();
};

DEVICE Fr_t ulong_to_scalar(unsigned long num) {
    return {static_cast<uint32_t>(num), static_cast<uint32_t>(num >> 32), 0, 0, 0, 0, 0, 0};
}

DEVICE unsigned long scalar_to_ulong(Fr_t num){
    return static_cast<unsigned long>(num.val[0]) | (static_cast<unsigned long>(num.val[1]) << 32);
}

DEVICE Fr_t relu_get_bit(unsigned long num, uint idx)
{
    return ((num >> idx) & 1) ? blstrs__scalar__Scalar_ONE: blstrs__scalar__Scalar_ZERO;
}



__global__ void relu_kernel(Fr_t* X, Fr_t* Z, Fr_t* sign, Fr_t* mag_bin, Fr_t* rem_bin, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    Fr_t x_unmont = blstrs__scalar__Scalar_unmont(X[gid]);
    Fr_t mag = x_unmont;
    if (blstrs__scalar__Scalar_gte({4294967295U, 32767U, 0U, 0U, 0U, 0U, 0U, 0U}, x_unmont))
    {
        sign[gid] = blstrs__scalar__Scalar_ZERO;
    }
    else if (blstrs__scalar__Scalar_gte(x_unmont, {1U, 4294934527U, 4294859774U, 1404937218U, 161601541U, 859428872U, 698187080U, 1944954707U}))
    {
        sign[gid] = blstrs__scalar__Scalar_ONE;
        mag = blstrs__scalar__Scalar_add(x_unmont, {0U, 32768U, 0U, 0U, 0U, 0U, 0U, 0U});
    }
    int rem = static_cast<uint>(x_unmont.val[0] & 65535U); // last 16 digits
    uint mag_rescaled = static_cast<uint>((scalar_to_ulong(mag) - rem) >> 16);

    #pragma unroll
    for(uint i = 0; i < 32; ++ i) mag_bin[gid * 32 + i] = relu_get_bit(mag_rescaled, i);

    #pragma unroll
    for(uint i = 0; i < 16; ++ i) rem_bin[gid * 16 + i] = relu_get_bit(rem, i);

    Z[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont({mag_rescaled, 0, 0, 0, 0, 0, 0, 0}), sign[gid]);
}


FrTensor zkReLU::operator()(const FrTensor& X) {
    reset_ptrs(X.size);
    FrTensor out(X.size);

    relu_kernel<<<(X.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(X.gpu_data, out.gpu_data, sign_ptr->gpu_data, mag_bin_ptr->gpu_data, rem_bin_ptr->gpu_data, X.size);
    cudaDeviceSynchronize();
    return out;
}

void zkReLU::reset_ptrs(uint size)
{
    if (sign_ptr) delete sign_ptr;
    if (mag_bin_ptr) delete mag_bin_ptr;
    if (rem_bin_ptr) delete rem_bin_ptr;

    sign_ptr = new FrTensor(size);
    mag_bin_ptr = new FrTensor(size * 32);
    rem_bin_ptr = new FrTensor(size * 16);
}

zkReLU::~zkReLU()
{
    if (sign_ptr) delete sign_ptr;
    if (mag_bin_ptr) delete mag_bin_ptr;
    if (rem_bin_ptr) delete rem_bin_ptr;
}


#endif  // ZKRELU_CUH
