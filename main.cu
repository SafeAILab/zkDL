#include "bls12-381.cuh"
#include <iostream>

using namespace std;

typedef blstrs__scalar__Scalar Fr;

int main()
{
  uint size = 1 << 10;
  Fr* gpu_data = nullptr;

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpu_data, sizeof(Fr) * size));
  
  cout << gpu_data << endl;

  return 0;
}