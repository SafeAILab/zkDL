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



// kernel to do float_to_fr conversion on a single float
__global__ void float_to_fr_kernel(float *f, Fr_t *out)
{
    *out = float_to_Fr(*f);
}

// kernel to do fr_to_float conversion on a single fr
__global__ void fr_to_float_kernel(Fr_t *f, float *out)
{
    *out = Fr_to_float(*f);
}


int main(int argc, char *argv[]) // float in
{   
    float *f;
    cudaMallocManaged(&f, sizeof(float));
    *f = std::stof(argv[1]);
    Fr_t *fr;
    // print out the float
    std::cout << "float: " << *f << std::endl;
    // put the float and the fr into device memory
    cudaMallocManaged(&fr, sizeof(Fr_t));
    float_to_fr_kernel<<<1, 1>>>(f, fr);
    cudaDeviceSynchronize();
    // print out the fr
    std::cout << "fr: " << *fr << std::endl;
    // now convert back to float
    float *f2;
    cudaMallocManaged(&f2, sizeof(float));
    fr_to_float_kernel<<<1, 1>>>(fr, f2);
    cudaDeviceSynchronize();
    // print out the float
    std::cout << "float: " << *f2 << std::endl;

}