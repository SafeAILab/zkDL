#ifndef PROOF_CUH
#define PROOF_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "bls12-381.cuh"

#include <vector>
#include <random>


vector<Fr_t> random_vec(uint len);
uint ceilLog2(uint num);

template<typename T>
std::vector<T> concatenate(const std::vector<std::vector<T>>& vecs);


KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size);

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof);

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u);

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v);

KERNEL void Fr_bin_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size);

void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);

vector<Fr_t> binary_sumcheck(const FrTensor& a, vector<Fr_t> u, vector<Fr_t> v);

#endif