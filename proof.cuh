#include "fr-tensor.cuh"
#include "g1-tensor.cuh"

#include <vector>

KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    Fr_t a0 = (gid0 < in_size) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < in_size) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < in_size) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < in_size) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_mul(a0, b0);
    out1[gid] = blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(a0, blstrs__scalar__Scalar_sub(b1, b0)), 
        blstrs__scalar__Scalar_mul(b0, blstrs__scalar__Scalar_sub(a1, a0)));
    out2[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_sub(a1, a0), blstrs__scalar__Scalar_sub(b1, b0));
}

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (begin >= end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    proof.push_back(out0.sum());
    proof.push_back(out1.sum());
    proof.push_back(out2.sum());

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_ip_sc(a_new, b_new, begin + 1, end, proof);
}

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u)
{
    vector<Fr_t> proof;
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size <= (1 << (log_size - 1))) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_ip_sc(a, b, u.begin(), u.end(), proof);
    return proof;
}

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_hp_sc(a_new, b_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions");
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size <= (1 << (log_size - 1))) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_hp_sc(a, b, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}