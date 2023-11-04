#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

#include <torch/torch.h>
#include <torch/script.h>

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "proof.cuh"

#include "timer.hpp"
#include "zkfc.cuh"
#include "zkrelu.cuh"
#include "zk-attention.cuh"

KERNEL void modular_inverse_kernel(Fr_t *input, uint size){
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        Fr_t m = blstrs__scalar__Scalar_mont(input[idx]);
        input[idx] = blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar_mul( modular_inverse(m), m));
    }
}



int main(int argc, char *argv[]) 
{   
    // // test the attention function
    // // V is n x d
    // // K is n x d
    // // Q is m x d
    // int n, m, d;
    // cin >> n >> m >> d;
    // Fr_t cpu_data[] = {{2,0,0,0,0,0,0,0},{8,0,0,0,0,0,0,0},{16,0,0,0,0,0,0,0},{4,0,0,0,0,0,0,0} };
    // auto V = FrTensor(n * d, cpu_data);
    // auto K = FrTensor(n * d, cpu_data);
    // auto Q = FrTensor(m * d, cpu_data);
    // auto out = FrTensor(m * d, cpu_data);
    // cout << V << endl;
    // cout << K << endl;
    // cout << Q << endl;
    // cout << out << endl << endl;
    // zkAttention::attention(V, K, Q, out, n, d, n , d,  m, d);
    // // print cuda errors
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    // }
    // cout << V << endl;
    // cout << K << endl;
    // cout << Q << endl;
    // cout << out << endl;
    // cout << blstrs__scalar__Scalar_P << endl;

    // test the modular_inverse function
    Fr_t *f;
    int size = 16;
    Fr_t f_cpu[size];
    f_cpu[0] = {1,1,3,1,2,4,2,1};
    // allocate gpu memory for f
    cudaMalloc(&f, size * sizeof(Fr_t));
    


    // call KERNEL void random_kernel(Fr_t* gpu_data, uint n, unsigned long seed) on f
    random_kernel<<<(size + 255) / 256, 256>>>(f, size, 0);
    cudaDeviceSynchronize();
    // read f out to cpu memory and print it
    cudaMemcpy(f_cpu, f, size * sizeof(Fr_t), cudaMemcpyDeviceToHost);  
    for (int i = 0; i < size; i++){
        cout << f_cpu[i] << endl;
    }
    // call KERNEL void modular_inverse_kernel(Fr_t *input, uint size) on f
    modular_inverse_kernel<<<(size + 255) / 256, 256>>>(f, size);
    cudaDeviceSynchronize();
    // read f out to cpu memory and print it
    cudaMemcpy(f_cpu, f, size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++){
        cout << f_cpu[i] << endl;
    }
}