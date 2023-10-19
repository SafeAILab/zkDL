#ifndef ZKFC_CUH
#define ZKFC_CUH

#include <torch/torch.h>
#include <torch/script.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"

#define TILE_WIDTH 16

class zkFC {
private:
    
    FrTensor weights;
    G1TensorJacobian com;

public:
    const uint inputSize;
    const uint outputSize;
    //zkFC(uint input_size, uint output_size, uint num_bits, const Commitment& generators);
    zkFC(uint input_size, uint output_size, const FrTensor& t, const Commitment& generators);
    FrTensor operator()(const FrTensor& X) const;
    void prove(const FrTensor& X, const FrTensor& Z, Commitment& generators) const;

    // static zkFC random_fc(uint input_size, uint output_size, uint num_bits, const Commitment& generators);
    static zkFC from_float_gpu_ptr (uint input_size, uint output_size, float* float_gpu_ptr, const Commitment& generators);
    static FrTensor load_float_gpu_input(uint batch_size, uint input_dim, float* input_ptr);
};

KERNEL void matrixMultiplyOptimized(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int colsB);

// KERNEL void random_init(Fr_t* params, uint num_bits, uint n)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     curandState state;
    
//     // Initialize the RNG state for this thread.
//     curand_init(1234, tid, 0, &state);  
    
//     if (tid < n) {
//         params[tid] = {curand(&state) & ((1U << num_bits) - 1), 0, 0, 0, 0, 0, 0, 0};
//         params[tid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_sub(params[tid], {1U << (num_bits - 1), 0, 0, 0, 0, 0, 0, 0}));
//     }
// }

DEVICE Fr_t float_to_Fr(float x);

KERNEL void float_to_Fr_kernel(float* fs, Fr_t* frs, uint fs_num_window, uint frs_num_window, uint fs_window_size, uint frs_window_size);

#endif  // ZKFC_CUH
