#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"

class Commitment: protected G1TensorJacobian
{
    using G1TensorJacobian::G1TensorJacobian;

    using G1TensorJacobian::operator+;
    using G1TensorJacobian::operator-;
    using G1TensorJacobian::operator*;

    G1TensorJacobian commit(const FrTensor& t);

    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u);
};

KERNEL void sum_axis_n_optimized(GLOBAL G1Jacobian_t* arr, GLOBAL G1Jacobian_t* arr_out, uint n, uint m) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m) return;
    
    __shared__ G1Jacobian_t shared_data[G1NumThread];
    uint local_id = threadIdx.x;
    
    G1Jacobian_t sum = arr[gid * n + local_id];
    for (uint i = local_id + G1NumThread; i < n; i += G1NumThread) {
        sum = blstrs__g1__G1Affine_add(sum, arr[gid * n + i]);
    }
    shared_data[local_id] = sum;
    
    __syncthreads();
    
    // Reduce shared_data to a single value
    for (uint s = G1NumThread / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_data[local_id] = blstrs__g1__G1Affine_add(shared_data[local_id], shared_data[local_id + s]);
        }
        __syncthreads();
    }
    
    if (local_id == 0) arr_out[gid] = shared_data[0];
}

G1TensorJacobian Commitment::commit(const FrTensor& scalar_tensor)
{
    if (scalar_tensor.size % size != 0) throw std::runtime_error("Incompatible dimensions");
    uint m = scalar_tensor.size / size;
    G1TensorJacobian temp = (*this) * t;
    G1TensorJacobian out(m);
    sum_axis_n_optimized<<<(m+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, out.gpu_data, n, m);
    cudaDeviceSynchronize();
    return out;
}

#endif