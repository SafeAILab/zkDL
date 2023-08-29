#include "fr-tensor.cuh"
#include "g1-tensor.cuh"

#include <vector>


// Matrix product of A (m x n) and B (n x p)
// Dimensions specified in log


FrTensor partial_reduce(const FrTensor&, const vector<Fr_t>&, uint window_size);

FrTensor partial_reduce(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size)
{
    if (begin >= end) return t;
    else
    {
        auto t_splitted = t.split(window_size);
        auto& t0 = t_splitted.first;
        auto& t1 = t_splitted.second;
        return partial_reduce(t0 + (t1 - t0) * (*begin), begin + 1, end, window_size);
    }
}

FrTensor partial_reduce(const FrTensor& t, const vector<Fr_t>& u, uint window_size)
{
    return partial_reduce(t, u.begin(), u.end(), window_size);
}

void matprod_proof(const FrTensor& A, const FrTensor& B, uint log_m, uint log_n, uint log_p, const vector<Fr_t>& vec_m, const vector<Fr_t>& vec_n, const vector<Fr_t>& vec_p)
{
    if (A.size != 1 << (log_m + log_n)) throw std::runtime_error("Incompatible dimensions.");
    if (B.size != 1 << (log_n + log_p)) throw std::runtime_error("Incompatible dimensions.");
    auto a = partial_reduce(A, vec_m, 1 << log_n);
    auto b = partial_reduce(B, vec_p, 1);
}