#ifndef FR_TENSOR_CUH
#define FR_TENSOR_CUH

#include <iostream>

#include "bls12-381.cuh"

typedef blstrs__scalar__Scalar Fr_t;
const uint NUM_THREAD = 1024;

// define the kernels

// Elementwise addition
KERNEL void Fr_elementwise_add(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr1[gid], arr2[gid]);
}

// Broadcast addition
KERNEL void Fr_broadcast_add(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr[gid], x);
}

// Elementwise negation
KERNEL void Fr_elementwise_neg(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ZERO, arr[gid]);
}

// Elementwise subtraction
KERNEL void Fr_elementwise_sub(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(arr1[gid], arr2[gid]);
}

// Broadcast subtraction
KERNEL void Fr_broadcast_sub(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(arr[gid], x);
}

// To montegomery form
KERNEL void Fr_elementwise_mont(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mont(arr[gid]);
}

// From montgomery form
KERNEL void Fr_elementwise_unmont(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_unmont(arr[gid]);
}

// Elementwise montegomery multiplication
KERNEL void Fr_elementwise_mont_mul(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mul(arr1[gid], arr2[gid]);
}

// Broadcast montegomery multiplication
KERNEL void Fr_broadcast_mont_mul(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mul(arr[gid], x);
}

KERNEL void Fr_sum_reduction(GLOBAL Fr_t *arr, GLOBAL Fr_t *output, int n) {
    extern __shared__ Fr_t sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < n) ? arr[i] : blstrs__scalar__Scalar_ZERO;
    if (i + blockDim.x < n) sdata[tid] = blstrs__scalar__Scalar_add(sdata[tid], arr[i + blockDim.x]);

    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = blstrs__scalar__Scalar_add(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to output
    if (tid == 0) output[blockIdx.x] = sdata[0];
}


// define the class FrTensor

class FrTensor
{   
    private:
    const uint size;
    Fr_t* gpu_data;

    public:
    FrTensor(uint size): size(size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
    }

    FrTensor(uint size, const Fr_t* cpu_data): size(size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
        cudaMemcpy(gpu_data, cpu_data, sizeof(Fr_t) * size, cudaMemcpyHostToDevice);
    }

    FrTensor(const FrTensor& t): size(t.size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
        cudaMemcpy(gpu_data, t.gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToDevice);
    }

    ~FrTensor()
    {
        cudaFree(gpu_data);
        gpu_data = nullptr;
    }

    Fr_t operator()(uint idx) const
    {
        Fr_t out;
        cudaMemcpy(&out, gpu_data + idx, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        return out;
    }

    FrTensor operator+(const FrTensor& t) const
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        FrTensor out(size);
        Fr_elementwise_add<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, t.gpu_data, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor operator+(const Fr_t& x) const
    {
        FrTensor out(size);
        Fr_broadcast_add<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, x, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor& operator+=(const FrTensor& t)
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        Fr_elementwise_add<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, t.gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }
    
    FrTensor& operator+=(const Fr_t& x)
    {
        Fr_broadcast_add<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, x, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }

    FrTensor operator-() const
    {
        FrTensor out(size);
        Fr_elementwise_neg<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor operator-(const FrTensor& t) const
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        FrTensor out(size);
        Fr_elementwise_sub<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, t.gpu_data, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor operator-(const Fr_t& x) const
    {
        FrTensor out(size);
        Fr_broadcast_sub<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, x, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor& operator-=(const FrTensor& t)
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        Fr_elementwise_sub<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, t.gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }
    
    FrTensor& operator-=(const Fr_t& x)
    {
        Fr_broadcast_sub<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, x, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }

    FrTensor& mont()
    {
        Fr_elementwise_mont<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    } 

    FrTensor& unmont()
    {
        Fr_elementwise_unmont<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    } 

    FrTensor operator*(const FrTensor& t) const
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        FrTensor out(size);
        Fr_elementwise_mont_mul<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, t.gpu_data, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor operator*(const Fr_t& x) const
    {
        FrTensor out(size);
        Fr_broadcast_mont_mul<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, x, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor& operator*=(const FrTensor& t)
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        Fr_elementwise_mont_mul<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, t.gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }
    
    FrTensor& operator*=(const Fr_t& x)
    {
        Fr_broadcast_mont_mul<<<(size+NUM_THREAD-1)/NUM_THREAD,NUM_THREAD>>>(gpu_data, x, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }
};

#endif