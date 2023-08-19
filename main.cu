#include "bls12-381.cuh"
#include "fr-tensor.cuh"
#include <iostream>
#include <iomanip>
#include "timer.hpp"

using namespace std;

// typedef blstrs__scalar__Scalar Fr;

// Fr randomFr()
// {
//   return {rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand() % 1944954707};
// }



// ostream& operator<<(ostream& os, const Fr& x)
// {
//   os << "0x" << std::hex;
//   for (uint i = 8; i > 0; -- i)
//   {
//     os << std::setfill('0') << std::setw(8) << x.val[i - 1];
//   }
//   return os << std::dec << std::setw(0) << std::setfill(' ');
// }

// KERNEL void Fr_add(GLOBAL Fr* arr1, GLOBAL Fr* arr2, GLOBAL Fr* arr_out, uint n) {
//   const uint gid = GET_GLOBAL_ID();
//   if (gid >= n) return;
//   arr_out[gid] = blstrs__scalar__Scalar_add(arr1[gid], arr2[gid]);
// }

// KERNEL void Fr_sub(GLOBAL Fr* arr1, GLOBAL Fr* arr2, GLOBAL Fr* arr_out, uint n) {
//   const uint gid = GET_GLOBAL_ID();
//   if (gid >= n) return;
//   arr_out[gid] = blstrs__scalar__Scalar_sub(arr1[gid], arr2[gid]);
// }

// KERNEL void Fr_bc_add(GLOBAL Fr* arr, Fr x, GLOBAL Fr* arr_out, uint n)
// {
//   const uint gid = GET_GLOBAL_ID();
//   if (gid >= n) return;
//   arr_out[gid] = blstrs__scalar__Scalar_add(arr[gid], x);
// }

// KERNEL void Fr_bc_sub(GLOBAL Fr* arr, Fr x, GLOBAL Fr* arr_out, uint n)
// {
//   const uint gid = GET_GLOBAL_ID();
//   if (gid >= n) return;
//   arr_out[gid] = blstrs__scalar__Scalar_sub(arr[gid], x);
// }



int main(int argc, char *argv[])
{
  uint nblock = stoi(argv[1]), nthread = stoi(argv[2]);
  uint size = nblock * nthread;

  Fr_t* cpu_data = new Fr_t[size];
  for (uint i = 0; i < size; ++ i)
  {
    cpu_data[i].val[7] = i;
    cpu_data[i].val[0] = size - i;
  }

  cout << "size=" << size << endl;
  Timer timer;
  timer.start();
  FrTensor t1(size);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  timer.start();
  FrTensor t2(size);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  timer.start();
  FrTensor t3(size, cpu_data);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  timer.start();
  FrTensor t4 = t3;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  delete[] cpu_data;
  
  return 0;

}

  //cout << cudaSetDevice(0) << endl;
  
  // Fr* cpu_data_1 = new Fr[size];
  // for (uint i = 0; i < size; ++ i)
  // {
  //   cpu_data_1[i].val[0] = i;
  //   // cout << cpu_data_1[i] << endl;
  // }

  // Fr* cpu_data_2 = new Fr[size];
  // for (uint i = 0; i < size; ++ i)
  // {
  //   cpu_data_2[i].val[0] = size - i;
  //   // cout << cpu_data_2[i] << endl;
  // }

  // Fr* gpu_data_1 = nullptr;
  // Fr* gpu_data_2 = nullptr;
  // Fr* gpu_data_3 = nullptr;
  // Fr* gpu_data_4 = nullptr;
  // Fr* gpu_data_5 = nullptr;
  // Fr* gpu_data_6 = nullptr;

  // Timer timer;

  // timer.start();
  // cudaMalloc((void **)&gpu_data_1, sizeof(Fr) * size);
  // timer.stop();
  // cout << timer.getTotalTime() << endl;
  // timer.reset();

  // timer.start();
  // cudaMalloc((void **)&gpu_data_2, sizeof(Fr) * size);
  // cudaMalloc((void **)&gpu_data_3, sizeof(Fr) * size);
  // cudaMalloc((void **)&gpu_data_4, sizeof(Fr) * size);
  // cudaMalloc((void **)&gpu_data_5, sizeof(Fr) * size);
  // cudaMalloc((void **)&gpu_data_6, sizeof(Fr) * size);
  // timer.stop();
  // cout << timer.getTotalTime() << endl;
  // timer.reset();

  // cudaMemcpy(gpu_data_1, cpu_data_1, sizeof(Fr) * size, cudaMemcpyHostToDevice);
  // cudaMemcpy(gpu_data_2, cpu_data_2, sizeof(Fr) * size, cudaMemcpyHostToDevice);

  // cout << "==== Kernel Tests Begin ====" << endl;
  
  // timer.start();
  // Fr_add<<<nblock, nthread>>>(gpu_data_1, gpu_data_2, gpu_data_3, size);
  // cout << cudaDeviceSynchronize() << endl;
  // timer.stop();
  // cout << timer.getTotalTime() << endl;
  // timer.reset();

  // timer.start();
  // Fr_sub<<<nblock, nthread>>>(gpu_data_1, gpu_data_2, gpu_data_4, size);
  // cout << cudaDeviceSynchronize() << endl;
  // timer.stop();
  // cout << timer.getTotalTime() << endl;
  // timer.reset();

  // timer.start();
  // Fr_bc_add<<<nblock, nthread>>>(gpu_data_1, {1123124, 2313124, 512523, 7547356, 743643, 132131, 8908907, 149892}, gpu_data_5, size);
  // cout << cudaDeviceSynchronize() << endl;
  // timer.stop();
  // cout << timer.getTotalTime() << endl;
  // timer.reset();
  
  // Fr x {625157, 5326, 598890, 141415, 6346346, 13515, 2463476, 23523};
  // timer.start();
  // Fr_bc_sub<<<nblock, nthread>>>(gpu_data_1, x, gpu_data_6, size);
  // cout << cudaDeviceSynchronize() << endl;
  // timer.stop();
  // cout << timer.getTotalTime() << endl;
  // timer.reset();

  // cout << "==== Kernel Tests End ====" << endl;

  // Fr* cpu_data_3 = new Fr[size];
  // Fr* cpu_data_4 = new Fr[size];
  // Fr* cpu_data_5 = new Fr[size];
  // Fr* cpu_data_6 = new Fr[size];

  // cout << cudaMemcpy(cpu_data_3, gpu_data_3, sizeof(Fr) * size, cudaMemcpyDeviceToHost) << endl;
  // cout << cudaMemcpy(cpu_data_4, gpu_data_4, sizeof(Fr) * size, cudaMemcpyDeviceToHost) << endl;
  // cout << cudaMemcpy(cpu_data_5, gpu_data_5, sizeof(Fr) * size, cudaMemcpyDeviceToHost) << endl;
  // cout << cudaMemcpy(cpu_data_6, gpu_data_6, sizeof(Fr) * size, cudaMemcpyDeviceToHost) << endl;

  // cout << cpu_data_3[0] << endl;
  // cout << cpu_data_4[0] << endl;
  // cout << cpu_data_5[1919] << endl;
  // cout << cpu_data_6[114514] << endl;

  // timer.start();
  // cudaFree(gpu_data_1);
  // cudaFree(gpu_data_2);
  // cudaFree(gpu_data_3);
  // cudaFree(gpu_data_4);
  // cudaFree(gpu_data_5);
  // cudaFree(gpu_data_6);
  // timer.stop();
  // cout << timer.getTotalTime() << endl;
  // timer.reset();

//   return 0;
// }