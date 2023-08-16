#include "bls12-381.cuh"
#include <iostream>
#include <iomanip>
#include "timer.hpp"

using namespace std;

typedef blstrs__scalar__Scalar Fr;

// Fr randomFr()
// {
//   return {rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand() % 1944954707};
// }

ostream& operator<<(ostream& os, const Fr& x)
{
  os << "0x" << std::hex;
  for (uint i = 8; i > 0; -- i)
  {
    os << std::setfill('0') << std::setw(8) << x.val[i - 1];
  }
  return os << std::dec << std::setw(0) << std::setfill(' ');
}

KERNEL void Fr_add(GLOBAL Fr* arr1, GLOBAL Fr* arr2, GLOBAL Fr* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr1[gid], arr2[gid]);
}

KERNEL void Fr_sub(GLOBAL Fr* arr1, GLOBAL Fr* arr2, GLOBAL Fr* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(arr1[gid], arr2[gid]);
}

// KERNEL void Fr_mul(GLOBAL Fr* arr1, GLOBAL Fr* arr2, GLOBAL Fr* arr_out, uint n) {
//   const uint gid = GET_GLOBAL_ID();
//   if (gid >= n) return;

//   arr_out[gid] = blstrs__scalar__Scalar_mul(arr1[gid], arr2[gid]);
// }

int main(int argc, char *argv[])
{
  uint nblock = stoi(argv[1]), nthread = stoi(argv[2]);
  uint size = nblock * nthread;

  cout << cudaSetDevice(0) << endl;
  
  Fr* cpu_data_1 = new Fr[size];
  for (uint i = 0; i < size; ++ i)
  {
    cpu_data_1[i].val[0] = i;
    // cout << cpu_data_1[i] << endl;
  }
  Fr* gpu_data_1 = nullptr;

  Fr* cpu_data_2 = new Fr[size];
  for (uint i = 0; i < size; ++ i)
  {
    cpu_data_2[i].val[0] = size - i;
    // cout << cpu_data_2[i] << endl;
  }
  Fr* gpu_data_2 = nullptr;

  Fr* gpu_data_3 = nullptr;
  Fr* gpu_data_4 = nullptr;

  cout << cudaMalloc((void **)&gpu_data_1, sizeof(Fr) * size) << endl;
  cout << cudaMalloc((void **)&gpu_data_2, sizeof(Fr) * size) << endl;
  cout << cudaMalloc((void **)&gpu_data_3, sizeof(Fr) * size) << endl;
  cout << cudaMalloc((void **)&gpu_data_4, sizeof(Fr) * size) << endl;
  
  cout << cudaMemcpy(gpu_data_1, cpu_data_1, sizeof(Fr) * size, cudaMemcpyHostToDevice) << endl;
  cout << cudaMemcpy(gpu_data_2, cpu_data_2, sizeof(Fr) * size, cudaMemcpyHostToDevice) << endl;
  
  Timer timer;
  timer.start();
  Fr_add<<<nblock, nthread>>>(gpu_data_1, gpu_data_2, gpu_data_3, size);
  cudaDeviceSynchronize();
  timer.stop();

  timer.start();
  Fr_sub<<<nblock, nthread>>>(gpu_data_1, gpu_data_2, gpu_data_4, size);
  cudaDeviceSynchronize();
  timer.stop();

  cout << timer.getTotalTime() << endl;

  Fr* cpu_data_3 = new Fr[size];
  Fr* cpu_data_4 = new Fr[size];

  cout << cudaMemcpy(cpu_data_3, gpu_data_3, sizeof(Fr) * size, cudaMemcpyDeviceToHost) << endl;
  cout << cudaMemcpy(cpu_data_4, gpu_data_4, sizeof(Fr) * size, cudaMemcpyDeviceToHost) << endl;

  cudaFree(gpu_data_1);
  cudaFree(gpu_data_2);
  cudaFree(gpu_data_3);
  cudaFree(gpu_data_4);

  delete[] cpu_data_1;
  delete[] cpu_data_2;
  delete[] cpu_data_3;
  delete[] cpu_data_4;

  return 0;
}