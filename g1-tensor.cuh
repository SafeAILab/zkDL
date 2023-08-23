#ifndef G1_TENSOR_CUH
#define G1_TENSOR_CUH

#include <iostream>

#include "bls12-381.cuh"

typedef blstrs__fp__Fp Fp_t;
const uint G1NumThread = 64;

typedef blstrs__g1__G1Affine_affine G1Affine_t;
typedef blstrs__g1__G1Affine_jacobian G1Jacobian_t;
const uint G1AffineSharedMemorySize = 2 * sizeof(G1Affine_t) * G1NumThread; 
const uint G1JacobianSharedMemorySize = 2 * sizeof(G1Jacobian_t) * G1NumThread;

DEVICE Fp_t Fp_minus(Fp_t a) {
  return blstrs__fp__Fp_sub(blstrs__fp__Fp_ZERO, a);
}

DEVICE G1Affine_t G1Affine_minus(G1Affine_t a) {
  return {a.x, Fp_minus(a.y)};
}

DEVICE G1Jacobian_t G1Jacobian_minus(G1Jacobian_t a) {
  return {a.x, Fp_minus(a.y), a.z};
}

// x_mont = 0x120177419e0bfb75edce6ecc21dbf440f0ae6acdf3d0e747154f95c7143ba1c17817fc679976fff55cb38790fd530c16
const Fp_t G1_generator_x_mont = {
    4250078230,
    1555269520,
    2574712821,
    2014837863,
    339452353,
    357537223,
    4090554183,
    4037962445,
    568063040,
    3989728972,
    2651585397,
    302085953
};

// y_mont = 0xbbc3efc5008a26a0e1c8c3fad0059c051ac582950405194dd595f13570725ce8c22631a7918fd8ebaac93d50ce72271
const Fp_t G1_generator_y_mont = {
    216474225,
    3131872213,
    2031680910,
    2351063834,
    1460086222,
    3713621779,
    1346392468,
    1370249257,
    2902481344,
    236751935,
    1342743146,
    196886268
};

const G1Affine_t G1Affine_generator {G1_generator_x_mont, G1_generator_y_mont};
const G1Jacobian_t G1Jacobian_generator {G1_generator_x_mont, G1_generator_y_mont, blstrs__fp__Fp_ONE};

class G1Tensor
{
    protected:
    const uint size;

    G1Tensor(uint size): size(size) {}
};

class G1TensorAffine: private G1Tensor
{
    private:
    G1Affine_t* gpu_data;

    public: 
    G1TensorAffine(const G1TensorAffine&);

    G1TensorAffine(uint size);

    G1TensorAffine(uint size, const G1Affine_t&);

    G1TensorAffine(uint size, const G1Affine_t* cpu_data);

    ~G1TensorAffine();

    G1TensorAffine operator-() const;

    friend class G1TensorJacobian;
};

class G1TensorJacobian: private G1Tensor
{
    private:
    G1Jacobian_t* gpu_data;

    public: 
    G1TensorJacobian(const G1TensorJacobian&);

    G1TensorJacobian(uint size);

    G1TensorJacobian(uint size, const G1Jacobian_t&);

    G1TensorJacobian(uint size, const G1Jacobian_t* cpu_data);

    G1TensorJacobian(const G1TensorAffine& affine_tensor);

    ~G1TensorJacobian();

    G1TensorJacobian operator-() const;

    G1TensorJacobian operator+(const G1TensorJacobian&) const;
    
    G1TensorJacobian operator+(const G1TensorAffine&) const;

    G1TensorJacobian operator+(const G1Jacobian_t&) const;

    G1TensorJacobian operator+(const G1Affine_t&) const;

    G1TensorJacobian& operator+=(const G1TensorJacobian&);
    
    G1TensorJacobian& operator+=(const G1TensorAffine&);

    G1TensorJacobian& operator+=(const G1Jacobian_t&);

    G1TensorJacobian& operator+=(const G1Affine_t&);

    G1TensorJacobian operator-(const G1TensorJacobian&) const;
    
    G1TensorJacobian operator-(const G1TensorAffine&) const;

    G1TensorJacobian operator-(const G1Jacobian_t&) const;

    G1TensorJacobian operator-(const G1Affine_t&) const;

    G1TensorJacobian& operator-=(const G1TensorJacobian&);
    
    G1TensorJacobian& operator-=(const G1TensorAffine&);

    G1TensorJacobian& operator-=(const G1Jacobian_t&);

    G1TensorJacobian& operator-=(const G1Affine_t&);

    friend class G1TensorAffine;
};

// Implement G1Affine

G1TensorAffine::G1TensorAffine(const G1TensorAffine& t): G1Tensor(t.size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Affine_t) * size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(G1Affine_t) * size, cudaMemcpyDeviceToDevice);
}

G1TensorAffine::G1TensorAffine(uint size): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Affine_t) * size);
}

KERNEL void G1Affine_assign_broadcast(GLOBAL G1Affine_t* arr, G1Affine_t g, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr[gid] = g;
}

G1TensorAffine::G1TensorAffine(uint size, const G1Affine_t& g): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Affine_t) * size);
    G1Affine_assign_broadcast<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, g, size);
    cudaDeviceSynchronize();
}

G1TensorAffine::G1TensorAffine(uint size, const G1Affine_t* cpu_data): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Affine_t) * size);
    cudaMemcpy(gpu_data, cpu_data, sizeof(G1Affine_t) * size, cudaMemcpyHostToDevice);
}

G1TensorAffine::~G1TensorAffine()
{
    cudaFree(gpu_data);
    gpu_data = nullptr;
}

KERNEL void G1_affine_elementwise_minus(GLOBAL G1Affine_t* arr_in, G1Affine_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = {arr_in[gid].x, blstrs__fp__Fp_sub(blstrs__fp__Fp_ZERO, arr_in[gid].y)};
}

G1TensorAffine G1TensorAffine::operator-() const
{
    G1TensorAffine out(size);
    G1_affine_elementwise_minus<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, out.gpu_data, size);
    cudaDeviceSynchronize();
    return out;
}


// Implement G1TensorJacobian

G1TensorJacobian::G1TensorJacobian(const G1TensorJacobian& t): G1Tensor(t.size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(G1Jacobian_t) * size, cudaMemcpyDeviceToDevice);
}

G1TensorJacobian::G1TensorJacobian(uint size): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
}

G1TensorJacobian::G1TensorJacobian(uint size, const G1Jacobian_t* cpu_data): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
    cudaMemcpy(gpu_data, cpu_data, sizeof(G1Jacobian_t) * size, cudaMemcpyHostToDevice);
}

KERNEL void G1Jacobian_assign_broadcast(GLOBAL G1Jacobian_t* arr, G1Jacobian_t g, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr[gid] = g;
}

G1TensorJacobian::G1TensorJacobian(uint size, const G1Jacobian_t& g): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
    G1Jacobian_assign_broadcast<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, g, size);
    cudaDeviceSynchronize();
}

KERNEL void G1_affine_to_jacobian(GLOBAL G1Affine_t* arr_affine, G1Jacobian_t* arr_jacobian, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_jacobian[gid] = {arr_affine[gid].x, arr_affine[gid].y, blstrs__fp__Fp_ONE};
}

G1TensorJacobian::G1TensorJacobian(const G1TensorAffine& affine_tensor): G1Tensor(affine_tensor.size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
    G1_affine_to_jacobian<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(affine_tensor.gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
}

G1TensorJacobian::~G1TensorJacobian()
{
    cudaFree(gpu_data);
    gpu_data = nullptr;
}

KERNEL void G1_jacobian_elementwise_minus(GLOBAL G1Jacobian_t* arr_in, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = {arr_in[gid].x, blstrs__fp__Fp_sub(blstrs__fp__Fp_ZERO, arr_in[gid].y), arr_in[gid].z};
}

G1TensorJacobian G1TensorJacobian::operator-() const
{
    G1TensorJacobian out(size);
    G1_jacobian_elementwise_minus<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, out.gpu_data, size);
    cudaDeviceSynchronize();
    return out;
}

KERNEL void G1_jacobian_elementwise_add(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Jacobian_t* arr2, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__g1__G1Affine_add(arr1[gid], arr2[gid]);
}

KERNEL void G1_jacobian_broadcast_add(GLOBAL G1Jacobian_t* arr, G1Jacobian_t x, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__g1__G1Affine_add(arr[gid], x);
}

KERNEL void G1_jacobian_elementwise_madd(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Affine_t* arr2, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__g1__G1Affine_add_mixed(arr1[gid], arr2[gid]);
}

KERNEL void G1_jacobian_broadcast_madd(GLOBAL G1Jacobian_t* arr, G1Affine_t x, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__g1__G1Affine_add_mixed(arr[gid], x);
}

KERNEL void G1_jacobian_elementwise_sub(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Jacobian_t* arr2, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__g1__G1Affine_add(arr1[gid], G1Jacobian_minus(arr2[gid]));
}

KERNEL void G1_jacobian_broadcast_sub(GLOBAL G1Jacobian_t* arr, G1Jacobian_t x, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__g1__G1Affine_add(arr[gid], G1Jacobian_minus(x));
}

KERNEL void G1_jacobian_elementwise_msub(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Affine_t* arr2, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__g1__G1Affine_add_mixed(arr1[gid], G1Affine_minus(arr2[gid]));
}

KERNEL void G1_jacobian_broadcast_msub(GLOBAL G1Jacobian_t* arr, G1Affine_t x, G1Jacobian_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__g1__G1Affine_add_mixed(arr[gid], G1Affine_minus(x));
}

#endif