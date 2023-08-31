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

    vector<Fr_t> u_A;
    u_A.insert(u_A.end(), u_n.begin(), u_n.end());
    u_A.insert(u_A.end(), u_m.begin(), u_m.end());
    vector<Fr_t> u_B;
    u_B.insert(u_B.end(), u_p.begin(), u_p.end());
    u_B.insert(u_B.end(), u_n.begin(), u_n.end());


    Timer timer;
    timer.start();
    auto a = A.partial_me(u_m, n);
    auto b = B.partial_me(u_p, 1);
    timer.stop();
    cout << timer.getTotalTime() << endl;
    timer.reset();

    timer.start();
    auto proof = inner_product_sumcheck(a, b, u_n);
    timer.stop();
    cout << timer.getTotalTime() << endl;
    timer.reset();

    timer.start();
    auto y_A = A(u_A);
    auto y_B = B(u_B);
    timer.stop();
    cout << timer.getTotalTime() << endl;
    cout << y_A << "\t" << a(u_n) << "\t" << proof[proof.size() - 2] << endl;
    cout << y_B << "\t" << b(u_n) << "\t" << proof[proof.size() - 1] << endl;
    timer.reset();

	delete[] cpu_data_A;
    delete[] cpu_data_B;
    cout << "Current CUDA status: " << cudaGetLastError() << endl;
	return 0;
}