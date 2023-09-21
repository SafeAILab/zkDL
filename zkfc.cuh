#ifndef ZKFC_CUH
#define ZKFC_CUH

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
    const uint inputSize;
    const uint outputSize;
    FrTensor weights;
    G1TensorJacobian com;

public:
    //zkFC(uint input_size, uint output_size, uint num_bits, const Commitment& generators);
    zkFC(uint input_size, uint output_size, const FrTensor& t, const Commitment& generators);
    FrTensor operator()(const FrTensor& X) const;
    void prove(const FrTensor& X, const FrTensor& Z) const;
    static zkFC random_fc(uint input_size, uint output_size, uint num_bits, const Commitment& generators);
};

__global__ void matrixMultiplyOptimized(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int colsB) {
    __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    Fr_t sum = blstrs__scalar__Scalar_ZERO;
    
    // Loop over the tiles of A and B required to compute the block sub-matrix
    for (int t = 0; t < (colsA - 1)/TILE_WIDTH + 1; ++t) {

        // Load the matrices from device memory to shared memory; each thread loads
        // one element of each matrix
        if (row < rowsA && t*TILE_WIDTH + threadIdx.x < colsA) {
            A_tile[threadIdx.y][threadIdx.x] = A[row*colsA + t*TILE_WIDTH + threadIdx.x];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
        }
        
        if (t*TILE_WIDTH + threadIdx.y < colsA && col < colsB) {
            B_tile[threadIdx.y][threadIdx.x] = B[(t*TILE_WIDTH + threadIdx.y)*colsB + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
        }

        // Synchronize to ensure all the data in shared memory is available
        __syncthreads();

        // Multiply the two matrices together;
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum = blstrs__scalar__Scalar_add(sum, blstrs__scalar__Scalar_mul(A_tile[threadIdx.y][k], B_tile[k][threadIdx.x]));
        }

        // Synchronize to ensure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row*colsB + col] = sum;
    }
}

KERNEL void random_init(Fr_t* params, uint num_bits, uint n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    
    // Initialize the RNG state for this thread.
    curand_init(1234, tid, 0, &state);  
    
    if (tid < n) {
        params[tid] = {curand(&state) & ((1U << num_bits) - 1), 0, 0, 0, 0, 0, 0, 0};
        params[tid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_sub(params[tid], {1U << (num_bits - 1), 0, 0, 0, 0, 0, 0, 0}));
    }
}

// zkFC::zkFC(uint input_size, uint output_size, uint num_bits, const Commitment& generators) : inputSize(input_size), outputSize(output_size), weights(input_size * output_size), com(input_size * output_size)
// {
//     random_init<<<(input_size*output_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(weights.gpu_data, num_bits, input_size * output_size);
//     cudaDeviceSynchronize();
// }

zkFC zkFC::random_fc(uint input_size, uint output_size, uint num_bits, const Commitment& c)
{
    FrTensor weights(input_size * output_size);
    random_init<<<(input_size*output_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(weights.gpu_data, num_bits, input_size * output_size);
    return zkFC(input_size, output_size, weights, c);
}

zkFC::zkFC(uint input_size, uint output_size, const FrTensor& t, const Commitment& c) : inputSize(input_size), outputSize(output_size), weights(t), com(c.commit(t)) {
    if (t.size != input_size * output_size) throw std::runtime_error("Incompatible dimensions");
}

FrTensor zkFC::operator()(const FrTensor& X) const {
    if (X.size % inputSize != 0) throw std::runtime_error("Incompatible dimensions");
    uint batchSize = X.size / inputSize;
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((outputSize + blockSize.x - 1) / blockSize.x, (batchSize + blockSize.y - 1) / blockSize.y);
    FrTensor out(batchSize * outputSize);
    matrixMultiplyOptimized<<<gridSize, blockSize>>>(X.gpu_data, weights.gpu_data, out.gpu_data, batchSize, inputSize, outputSize);
    cudaDeviceSynchronize();
    return out;
}

void zkFC::prove(const FrTensor& X, const FrTensor& Z, Commitment& generators) const {
    // cout << X.size << " " << inputSize << endl;
    if (X.size % inputSize != 0) {
        throw std::runtime_error("Incompatible dimensions 1");
    }
    uint batchSize = X.size / inputSize;
    // sumcheck for inner product
    auto u_bs = random_vec(ceilLog2(batchSize));
    auto u_in_dim = random_vec(ceilLog2(inputSize));
    auto u_out_dim = random_vec(ceilLog2(outputSize));
    // cout << u_bs.size() << " " << u_in_dim.size() << " " << u_out_dim.size() << endl;
    inner_product_sumcheck(X.partial_me(u_bs, inputSize), weights.partial_me(u_out_dim, 1), u_in_dim);
    // cout << "Inner product sumcheck success" << endl;
    auto u_Z = concatenate<Fr_t>({u_out_dim, u_bs});
    // cout << u_Z.size() << " " << Z.size << endl;
    Z(u_Z);
    generators.open(weights, com, );
}



#endif  // ZKFC_CUH
