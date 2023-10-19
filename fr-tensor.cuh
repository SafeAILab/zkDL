#ifndef FR_TENSOR_CUH
#define FR_TENSOR_CUH

#include <iostream>
#include <iomanip>
#include <utility>      // std::pair, std::make_pair
#include <vector>
#include <curand_kernel.h>
#include <random>
#include "bls12-381.cuh"
using namespace std;

typedef blstrs__scalar__Scalar Fr_t;
typedef blstrs__g1__G1Affine_affine G1Affine_t;
typedef blstrs__g1__G1Affine_jacobian G1Jacobian_t;

const uint FrNumThread = 256;
const uint FrSharedMemorySize = 2 * sizeof(Fr_t) * FrNumThread; 

ostream& operator<<(ostream& os, const Fr_t& x);



// define the kernels

// Elementwise addition
KERNEL void Fr_elementwise_add(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n);

// Broadcast addition
KERNEL void Fr_broadcast_add(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n);

// Elementwise negation
KERNEL void Fr_elementwise_neg(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n);

// Elementwise subtraction
KERNEL void Fr_elementwise_sub(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n);

// Broadcast subtraction
KERNEL void Fr_broadcast_sub(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n);

// To montegomery form
KERNEL void Fr_elementwise_mont(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n);

// From montgomery form
KERNEL void Fr_elementwise_unmont(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n);

// Elementwise montegomery multiplication
KERNEL void Fr_elementwise_mont_mul(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n);

// Broadcast montegomery multiplication
KERNEL void Fr_broadcast_mont_mul(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n);


class G1TensorAffine;
class G1TensorJacobian;
class Commitment;
class zkFC;
class zkReLU;

// define the class FrTensor

class FrTensor
{   
    private:
    Fr_t* gpu_data;

    public:
    const uint size;
    FrTensor(uint size);

    FrTensor(uint size, const Fr_t* cpu_data);

    FrTensor(const FrTensor& t);

    ~FrTensor();

    Fr_t operator()(uint idx) const;

    FrTensor operator+(const FrTensor& t) const;

    FrTensor operator+(const Fr_t& x) const;

    FrTensor& operator+=(const FrTensor& t);
    
    FrTensor& operator+=(const Fr_t& x);

    FrTensor operator-() const;

    FrTensor operator-(const FrTensor& t) const;

    FrTensor operator-(const Fr_t& x) const;

    FrTensor& operator-=(const FrTensor& t);
    
    FrTensor& operator-=(const Fr_t& x);

    FrTensor& mont();

    FrTensor& unmont();

    FrTensor operator*(const FrTensor& t) const;

    FrTensor operator*(const Fr_t& x) const;

    FrTensor& operator*=(const FrTensor& t);
    
    FrTensor& operator*=(const Fr_t& x);

    Fr_t sum() const;

    Fr_t operator()(const vector<Fr_t>& u) const;

    std::pair<FrTensor, FrTensor> split(uint window_size) const;

    FrTensor partial_me(vector<Fr_t> u, uint window_size) const;

    static FrTensor random_int(uint size, uint num_bits);
    static FrTensor random(uint size);

    friend Fr_t Fr_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end);

    friend FrTensor Fr_partial_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size);

    friend void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof);
    friend void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);
    friend void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);

    friend class G1TensorAffine;
    friend class G1TensorJacobian;
    friend class Commitment;
    friend class zkFC;
    friend class zkReLU;
};

KERNEL void Fr_sum_reduction(GLOBAL Fr_t *arr, GLOBAL Fr_t *output, uint n);


KERNEL void random_int_kernel(Fr_t* gpu_data, uint num_bits, uint n, unsigned long seed);


KERNEL void random_kernel(Fr_t* gpu_data, uint n, unsigned long seed);



KERNEL void Fr_split_by_window(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr0, GLOBAL Fr_t *arr1, uint in_size, uint out_size, uint window_size);

KERNEL void Fr_me_step(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint in_size, uint out_size);

KERNEL void Fr_partial_me_step(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint in_size, uint out_size, uint window_size);

ostream& operator<<(ostream& os, const FrTensor& A);

#endif