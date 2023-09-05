#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"
#include <iostream>
#include <iomanip>
#include <random>
#include "timer.hpp"
#include "zkfc.cuh"
#include "zkrelu.cuh"

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
    uint batch_size = stoi(argv[1]);
    uint dim_in = stoi(argv[2]);
    uint dim_out = stoi(argv[3]);

    uint rng = 1U << 20;

    FrTensor x = random_tensor(batch_size * dim_in, rng);
    FrTensor w = random_tensor(dim_in * dim_out, rng);
    zkFC fc(dim_in, dim_out, w);
    zkReLU relu;
    Timer timer;
    
    FrTensor sign(batch_size * dim_out);
    FrTensor mag_bin(batch_size * dim_out * 32);
    FrTensor rem_bin(batch_size * dim_out * 16);

    timer.start();
    auto z = relu(fc(x));
    timer.stop();

    cout << timer.getTotalTime() << endl;

    auto error = cudaGetLastError();
    if (error) {
        cout << "Current CUDA status: " << error << endl;
    }
	return error;
}