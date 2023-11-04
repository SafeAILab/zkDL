#ifndef ZK_ATTENTION_CUH
#define ZK_ATTENTION_CUH

#include <torch/torch.h>
#include <torch/script.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "zkfc.cuh"

class zkAttention{
    public:
    static void attention(FrTensor &V, FrTensor &K, FrTensor &Q, FrTensor &out, uint rowsV, uint colsV, uint rowsK, uint colsK, uint rowsQ, uint colsQ);

};

DEVICE float Fr_to_float(Fr_t f);

KERNEL void Fr_to_float_kernel(Fr_t* input, float* output,  uint rows, uint cols);

KERNEL void matrixMultiplyTranspose(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int rowsB);

KERNEL void softmax(float *A, float *P, int rows, int cols);

KERNEL void broadcastMultiplyFloat(float *A, float *C, int size, float scalar);

#endif
