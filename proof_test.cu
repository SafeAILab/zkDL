#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"
#include <iostream>
#include <iomanip>
#include <random>
#include "timer.hpp"
// #include "zkfc.cuh"

using namespace std;

vector<Fr_t> random_vec(uint len)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    vector<Fr_t> out(len);
    for (uint i = 0; i < len; ++ i) out[i] = {dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), 0};
    return out;
}

FrTensor random_tensor(uint size, uint range)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned int> dist(0, range - 1);
    Fr_t* pos = new Fr_t[size];
    Fr_t* neg = new Fr_t[size];
    for (uint i = 0; i < size; ++ i) 
    {
        pos[i] = {dist(mt), 0, 0, 0, 0, 0, 0, 0};
        neg[i] = {dist(mt), 0, 0, 0, 0, 0, 0, 0};
    }
    FrTensor pos_tensor(size, pos);
    FrTensor neg_tensor(size, neg);
    delete[] pos;
    delete[] neg;
    return (pos_tensor - neg_tensor).mont();
}

FrTensor random_binary_tensor(uint size)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned int> dist(0, 1);
    Fr_t* arr = new Fr_t[size];
    for (uint i = 0; i < size; ++ i) 
    {
        arr[i] = {dist(mt), 0, 0, 0, 0, 0, 0, 0};
    }
    FrTensor out(size, arr);
    delete[] arr;
    return out.mont();
}

int main(int argc, char *argv[])
{   
    uint log_bs = stoi(argv[1]);
    uint log_dim_in = stoi(argv[2]);
    uint log_dim_out = stoi(argv[3]);
    uint log_Q = 5;
    uint log_R = 4;

    uint bs = 1U << log_bs;
    uint dim_in = 1U << log_dim_in;
    uint dim_out = 1U << log_dim_out;
    uint Q = 1U << log_Q;
    uint R = 1U << log_R;

    uint rng = 1U << 20;

    FrTensor x = random_tensor(bs * dim_in, rng);
    FrTensor w = random_tensor(dim_in * dim_out, rng);

    Timer timer;
    
    // sumcheck for inner product
    auto u_bs = random_vec(log_bs);
    auto u_in_dim = random_vec(log_dim_in);
    auto u_out_dim = random_vec(log_dim_out);
    timer.start();
    inner_product_sumcheck(x.partial_me(u_bs, dim_in), w.partial_me(u_out_dim, 1), u_in_dim);
    timer.stop();

    // sumcheck for two binaries
    FrTensor z_bin = random_binary_tensor(bs * dim_out * Q);
    FrTensor r_bin = random_binary_tensor(bs * dim_out * R);
    auto u_z_bin = random_vec(log_bs + log_dim_out + log_Q);
    auto v_z_bin = random_vec(log_bs + log_dim_out + log_Q);
    auto u_r_bin = random_vec(log_bs + log_dim_out + log_R);
    auto v_r_bin = random_vec(log_bs + log_dim_out + log_R); 
    auto u_recover = random_vec(log_bs + log_dim_out);
    timer.start();
    binary_sumcheck(z_bin, u_z_bin, v_z_bin);
    z_bin.partial_me(u_recover, Q);
    binary_sumcheck(r_bin, u_r_bin, v_r_bin);
    r_bin.partial_me(u_recover, R);
    timer.stop();

    // sumcheck for relu forward
    FrTensor z = random_tensor(bs * dim_out, rng);
    FrTensor mask = random_tensor(bs * dim_out, rng);
    auto u_hp = random_vec(log_bs + log_dim_out);
    auto v_hp = random_vec(log_bs + log_dim_out);
    timer.start();
    hadamard_product_sumcheck(z, mask, u_hp, v_hp);
    timer.stop();

    cout << timer.getTotalTime() << endl;

    // zkFC fc(dim, dim);

    auto error = cudaGetLastError();
    if (error) {
        cout << "Current CUDA status: " << error << endl;
    }
	return error;
}