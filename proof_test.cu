#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"
#include <iostream>
#include <iomanip>
#include <random>
#include "timer.hpp"

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

int main(int argc, char *argv[])
{
	uint log_m = stoi(argv[1]);
    uint log_n = stoi(argv[2]);
	uint log_p = stoi(argv[3]);

    uint m = 1 << log_m;
    uint n = 1 << log_n;
    uint p = 1 << log_p;

	Fr_t* cpu_data_A = new Fr_t[m * n];
	for (uint i = 0; i < m; ++ i)
	{
        for (uint j = 0; j < n; ++ j)
        {
            cpu_data_A[i * n + j] = {0, 0, 0, i, 0, 0, 0, j};
        }
		
	}

    Fr_t* cpu_data_B = new Fr_t[n * p];
    for (uint i = 0; i < n; ++ i)
	{
        for (uint j = 0; j < p; ++ j)
        {
            cpu_data_B[i * p + j] = {0, 0, 0, i, 0, 0, 0, j};
        }
	}

    

    FrTensor A(m * n, cpu_data_A);
    FrTensor B(n * p, cpu_data_B);

    auto u_m = random_vec(log_m);
    auto u_n = random_vec(log_n);
    auto u_p = random_vec(log_p);

    Timer timer;
    timer.start();
    auto a = A.partial_me(u_m, n);
    auto b = B.partial_me(u_p, 1);
    timer.stop();

	cout << "Current CUDA status: " << cudaGetLastError() << endl;
    cout << a.size << "\t" << b.size << endl;
    cout << timer.getTotalTime() << endl;
    timer.reset();

	delete[] cpu_data_A;
    delete[] cpu_data_B;
	return 0;
}