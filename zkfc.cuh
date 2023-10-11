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

DEVICE Fr_t float_to_Fr(float x)
{
    x = x * (1 << 16);
    float abs_x = round(abs(x));
    float sign_x = copysign(1.0f, x);

    bool negative = (sign_x < 0);
    uint rounded_abs = static_cast<uint>(abs_x);

    if (negative){
        return blstrs__scalar__Scalar_sub({0, 0, 0, 0, 0, 0, 0, 0}, {rounded_abs, 0, 0, 0, 0, 0, 0, 0});
    }
    else {
        return {rounded_abs, 0, 0, 0, 0, 0, 0, 0};
    }
}

KERNEL void float_to_Fr_kernel(float* fs, Fr_t* frs, uint fs_num_window, uint frs_num_window, uint fs_window_size, uint frs_window_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint dim0 = tid / frs_window_size;
    uint dim1 = tid % frs_window_size;
    if (tid >= frs_num_window * frs_window_size) return;
    if (dim0 < fs_num_window && dim1 < fs_window_size) frs[dim0 * frs_window_size + dim1] = float_to_Fr(fs[dim0 * fs_window_size + dim1]);
    else frs[tid] = {0, 0, 0, 0, 0, 0, 0, 0};
}

zkFC zkFC::from_float_gpu_ptr (uint input_size, uint output_size, float* float_gpu_ptr, const Commitment& generators)
{   
    uint rounded_input_size = 1 << ceilLog2(input_size);
    uint rounded_output_size = 1 << ceilLog2(output_size);

    FrTensor weights(rounded_input_size * rounded_output_size);
    float_to_Fr_kernel<<<(rounded_input_size * rounded_output_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(float_gpu_ptr, weights.gpu_data, input_size, rounded_input_size, output_size, rounded_output_size);
    cudaDeviceSynchronize();
    // cout << "Loaded weight is: " << weights << endl;
    return zkFC(rounded_input_size, rounded_output_size, weights.mont(), generators);
}

zkFC::zkFC(uint input_size, uint output_size, const FrTensor& t, const Commitment& c) : inputSize(input_size), outputSize(output_size), weights(t), com(c.commit(t)) {
    if (t.size != input_size * output_size) throw std::runtime_error("Incompatible dimensions");
}

FrTensor zkFC::load_float_gpu_input(uint batch_size, uint input_dim, float* input_ptr)
{
    uint rounded_batch_size = 1 << ceilLog2(batch_size);
    uint rounded_input_dim = 1 << ceilLog2(input_dim);
    FrTensor t(rounded_batch_size * rounded_input_dim);
    float_to_Fr_kernel<<<(rounded_batch_size * rounded_input_dim+FrNumThread-1)/FrNumThread,FrNumThread>>>(input_ptr, t.gpu_data, batch_size, rounded_batch_size, input_dim, rounded_input_dim);
    cudaDeviceSynchronize();
    // cout << "Loaded input is: " << t << endl;
    return t;
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
    generators.open(weights, com, concatenate<Fr_t>({u_out_dim, u_in_dim}));
}



#endif  // ZKFC_CUH
