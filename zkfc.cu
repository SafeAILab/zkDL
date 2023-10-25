#include "zkfc.cuh"

#define TILE_WIDTH 16

// rudimentary tiled softmax on a float matrix
KERNEL void softmax(float *A, float *P, int rows, int cols) {
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y;
    int col = threadIdx.x;

    float sum = 0.0f;

    // Load the row of A from device memory to shared memory
    if (col < cols) {
        A_tile[threadIdx.y][threadIdx.x] = A[row*cols + col];
    } else {
        A_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Synchronize to ensure all the data in shared memory is available
    __syncthreads();

    // Compute the sum of the exponentiated elements of the row
    for (int i = 0; i < cols; ++i) {
        sum += expf(A_tile[threadIdx.y][i]);
    }

    // Compute the softmax of each element of the row
    for (int i = 0; i < cols; ++i) {
        P[row*cols + i] = expf(A_tile[threadIdx.y][i]) / sum;
    }
}

KERNEL void broadcastMultiplyFloat(float *A, float *C, int rows, int cols, float scalar) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        C[row*cols + col] = A[row*cols + col] * scalar;
    }
}

KERNEL void matrixMultiplyOptimized(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int colsB) {
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

// C = A * B^T
KERNEL void matrixMultiplyTranspose(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int rowsB) {
    __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    Fr_t sum = blstrs__scalar__Scalar_ZERO;
    
    // Loop over the tiles of A and B required to compute the block sub-matrix
    for (int t = 0; t < (rowsB - 1)/TILE_WIDTH + 1; ++t) {

        // Load the matrices from device memory to shared memory; each thread loads
        // one element of each matrix
        if (row < rowsA && t*TILE_WIDTH + threadIdx.y < rowsB) {
            A_tile[threadIdx.y][threadIdx.x] = A[row*colsA + threadIdx.x];
            B_tile[threadIdx.y][threadIdx.x] = B[(t*TILE_WIDTH + threadIdx.y)*colsA + threadIdx.x];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
            B_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
        }

        // Synchronize to ensure all the data in shared memory is available
        __syncthreads();

        // Multiply the two matrices together;
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum = blstrs__scalar__Scalar_add(sum, blstrs__scalar__Scalar_mul(A_tile[threadIdx.y][k], B_tile[threadIdx.x][k]));
        }

        // Synchronize to ensure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    if (row < rowsA && col < rowsB) {
        C[row*rowsB + col] = sum;
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

// the reverse of float_to_Fr
DEVICE float Fr_to_float(Fr_t f)
{
    // first set the sign flag by seeing if f[7] is non-zero in which case negative = true
    bool negative = (f.val[7] != 0);
    // if negative then take sub from 0
    if (negative) f = blstrs__scalar__Scalar_sub({0, 0, 0, 0, 0, 0, 0, 0}, f);

    // now f is positive
    float x = static_cast<float>(f.val[0]);
    // now divide by the scaling factor (1 << 16)
    x = x / (1 << 16);
    // now set the sign
    if (negative) x = -x; 
    return x;
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

// reverse of float_to_Fr_kernel
KERNEL void Fr_to_float_kernel(float* fs, Fr_t* frs, uint fs_num_window, uint frs_num_window, uint fs_window_size, uint frs_window_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint dim0 = tid / fs_window_size;
    uint dim1 = tid % fs_window_size;
    if (tid >= fs_num_window * fs_window_size) return;
    if (dim0 < frs_num_window && dim1 < frs_window_size) fs[dim0 * fs_window_size + dim1] = Fr_to_float(frs[dim0 * frs_window_size + dim1]);
    else fs[tid] = 0;
}

// Attention(V,K,Q) = softmax(QK^T/sqrt(d_k))V
// V, K, Q are on gpu
void zkFC::attention(FrTensor &V, FrTensor &K, FrTensor &Q, FrTensor &out, uint rowsV, uint colsV, uint rowsK, uint colsK, uint rowsQ, uint colsQ){
    // create QK
    FrTensor QK(rowsQ * rowsK);
    // calculate QK^T
    Q.mont();
    K.mont();
    // call matrixMultiplyTranspose
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((rowsQ + blockSize.x - 1) / blockSize.x, (rowsK + blockSize.y - 1) / blockSize.y);
    matrixMultiplyTranspose<<<gridSize, blockSize>>>(Q.gpu_data, K.gpu_data, QK.gpu_data, rowsQ, colsQ, rowsK);
    cudaDeviceSynchronize();
    // unmont
    QK.unmont();
    uint d = colsV;
    // convert QK to float
    // allocate gpu memory
    float *QK_float;
    cudaMalloc(&QK_float, rowsQ * rowsK * sizeof(float));
    // call Fr_to_float_kernel
    Fr_to_float_kernel<<<(rowsQ * rowsK + FrNumThread - 1) / FrNumThread, FrNumThread>>>(QK_float, QK.gpu_data, rowsQ, rowsQ, rowsK, rowsK);
    // broadcast multiply by inverse of sqrt(d)
    broadcastMultiplyFloat<<<(rowsQ * rowsK + FrNumThread - 1) / FrNumThread, FrNumThread>>>(QK_float, QK_float, rowsQ, rowsK, 1.0f / sqrtf(float(d)));
    // apply softmax
    // allocate memory for a softmaxed QK/ sqrt(d)
    float *QK_softmax;
    cudaMalloc(&QK_softmax, rowsQ * rowsK * sizeof(float));
    // call softmax
    softmax<<<rowsQ, colsK>>>(QK_float, QK_softmax, rowsQ, rowsK);
    // convert QK_softmax back to Fr
    float_to_Fr_kernel<<<(rowsQ * rowsK + FrNumThread - 1) / FrNumThread, FrNumThread>>>(QK_softmax, QK.gpu_data, rowsQ, rowsQ, rowsK, rowsK);
    // calculate QK^T * V
    // mont
    V.mont(); QK.mont();
    // call matrixMultiplyOptimized and write final result to out.gpu_data
    // calculate new gridSize and blockSize
    blockSize = dim3(TILE_WIDTH, TILE_WIDTH);
    gridSize = dim3((rowsQ + blockSize.x - 1) / blockSize.x, (colsV + blockSize.y - 1) / blockSize.y);
    // calculate rowsQK^T and colsQK^T
    uint rowsQK = rowsQ;
    uint colsQK = rowsK;
    // call matrixMultiplyOptimized
    matrixMultiplyOptimized<<<gridSize, blockSize>>>(QK.gpu_data, V.gpu_data, out.gpu_data, rowsQK, colsQK, colsV);
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

