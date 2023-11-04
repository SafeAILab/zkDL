#include "zk-attention.cuh"

// rudimentary tiled softmax on a float matrix
KERNEL void softmax(float *A, float *P, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            sum += (A[idx * cols + i]);
        }

        for (int i = 0; i < cols; ++i) {
            P[idx * cols + i] = (A[idx * cols + i]) / sum;
        }
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

// reverse of float_to_Fr_kernel
KERNEL void Fr_to_float_kernel(Fr_t* input, float* output,  uint rows, uint cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < rows * cols) {
        output[idx] = Fr_to_float(input[idx]);
    }
}


// C = AB^T
KERNEL void matrixMultiplyTranspose(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int rowsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int colsB = colsA;

    if (row < rowsA && col < rowsB) {
        Fr_t sum = blstrs__scalar__Scalar_ZERO;
        for (int i = 0; i < colsA; i++) {
            sum = blstrs__scalar__Scalar_add(sum, blstrs__scalar__Scalar_mul(A[row * colsA + i], B[col * colsB + i]));
        }
        C[row * rowsB + col] = sum;
    }
}

// C = A * B^T
// KERNEL void matrixMultiplyTranspose(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int rowsB) {
//     __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
//     __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];

//     int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
//     int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

//     Fr_t sum = blstrs__scalar__Scalar_ZERO;
    
//     // Loop over the tiles of A and B required to compute the block sub-matrix
//     for (int t = 0; t < (rowsB - 1)/TILE_WIDTH + 1; ++t) {

//         // Load the matrices from device memory to shared memory; each thread loads
//         // one element of each matrix
//         if (row < rowsA && t*TILE_WIDTH + threadIdx.y < rowsB) {
//             A_tile[threadIdx.y][threadIdx.x] = A[row*colsA + threadIdx.x];
//             B_tile[threadIdx.y][threadIdx.x] = B[(t*TILE_WIDTH + threadIdx.y)*colsA + threadIdx.x];
//         } else {
//             A_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
//             B_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
//         }

//         // Synchronize to ensure all the data in shared memory is available
//         __syncthreads();

//         // Multiply the two matrices together;
//         for (int k = 0; k < TILE_WIDTH; ++k) {
//             sum = blstrs__scalar__Scalar_add(sum, blstrs__scalar__Scalar_mul(A_tile[threadIdx.y][k], B_tile[threadIdx.x][k]));
//         }

//         // Synchronize to ensure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
//         __syncthreads();
//     }

//     if (row < rowsA && col < rowsB) {
//         C[row*rowsB + col] = sum;
//     }
// }


KERNEL void broadcastMultiplyFloat(float *A, float *C, int size, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}

// Attention(V,K,Q) = softmax(QK^T/sqrt(d_k))V
// V, K, Q have data on gpu
// inputs are not in mont, output is also not in mont
void zkAttention::attention(FrTensor &V, FrTensor &K, FrTensor &Q, FrTensor &out, uint rowsV, uint colsV, uint rowsK, uint colsK, uint rowsQ, uint colsQ){
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
    std::cout << "QK" << QK << std::endl;
    uint d = colsV;
    // convert QK to float
    // allocate gpu memory
    float *QK_float, *QK_float2;
    cudaMalloc(&QK_float, rowsQ * rowsK * sizeof(float));
    cudaMalloc(&QK_float2, rowsQ * rowsK * sizeof(float));
    // call Fr_to_float_kernel
    Fr_to_float_kernel<<<(rowsQ * rowsK + FrNumThread - 1) / FrNumThread, FrNumThread>>>(QK.gpu_data, QK_float, rowsQ, rowsK);
    cudaDeviceSynchronize();

    float *QK_float_cpu = (float *)malloc(rowsQ * rowsK * sizeof(float));
    // apply softmax
    // allocate memory for a softmaxed QK/ sqrt(d)
    float *QK_softmax;
    cudaMalloc(&QK_softmax, rowsQ * rowsK * sizeof(float));

    cudaMemcpy(QK_float_cpu, QK_float, rowsQ * rowsK * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "QK_float" << endl;
    for(uint i = 0; i < rowsQ; i++){
        for(uint j = 0; j < rowsK; j++){
            cout << QK_float_cpu[i * rowsK + j] << " ";
        }
        cout << endl;
    }

    cout << 1.0f / sqrtf(float(d)) << endl;

    // broadcast multiply by inverse of sqrt(d)
    broadcastMultiplyFloat<<<1, FrNumThread>>>(QK_float, QK_float2, rowsQ * rowsK, 1.0f / sqrtf(float(d)));
    cudaDeviceSynchronize();
    cudaFree(QK_float);
    QK_float = QK_float2;

    cudaMemcpy(QK_float_cpu, QK_float, rowsQ * rowsK * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "QK_float" << endl;
    for(uint i = 0; i < rowsQ; i++){
        for(uint j = 0; j < rowsK; j++){
            cout << QK_float_cpu[i * rowsK + j] << " ";
        }
        cout << endl;
    }

    softmax<<<2, 1>>>(QK_float, QK_softmax, rowsQ, rowsK);
    cudaDeviceSynchronize();
    // get QK_float back to cpu memory and print it out
    cudaMemcpy(QK_float_cpu, QK_softmax, rowsQ * rowsK * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "QK_float" << endl;
    for(uint i = 0; i < rowsQ; i++){
        for(uint j = 0; j < rowsK; j++){
            cout << QK_float_cpu[i * rowsK + j] << " ";
        }
        cout << endl;
    }
    cudaDeviceSynchronize();
    // convert QK_softmax back to Fr
    float_to_Fr_kernel<<<(rowsQ * rowsK + FrNumThread - 1) / FrNumThread, FrNumThread>>>(QK_softmax, QK.gpu_data, rowsQ, rowsQ, rowsK, rowsK);
    cudaDeviceSynchronize();
    // calculate QK^T * V
    // mont
    cout << "QK after softmax" << QK << endl;
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
    cudaDeviceSynchronize();
    out.unmont();
}

