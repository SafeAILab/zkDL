#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
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
	// uint size = stoi(argv[1]);
	// uint window_size = stoi(argv[2]);
	// bool need_print = false;
	// if (argc > 3) need_print = stoi(argv[3]);

	
	uint log_size_out = stoi(argv[1]);
    uint log_size_in = stoi(argv[2]);

    uint size_out = 1 << log_size_out;
    uint size_in = 1 << log_size_in;
    Commitment generators(size_in, G1Jacobian_generator);

    auto rnd_vec = random_vec(size_in);
    FrTensor rnd_tensor(size_in, &(rnd_vec.front()));
    generators *= rnd_tensor;

	Fr_t* cpu_data = new Fr_t[size_out * size_in];
	for (uint i = 0; i < size_out * size_in; ++ i)
	{
		cpu_data[i] = {i, 0, 0, 0, 0, 0, 0, 0};
	}
    Timer timer;
    
    FrTensor data_tensor(size_out * size_in, cpu_data);
    data_tensor.mont();

    timer.start();
    auto c = generators.commit(data_tensor);
    cout << "Current CUDA status: " << cudaGetLastError() << endl;
    timer.stop();
    cout << timer.getTotalTime() << endl;
    timer.reset();
	
    auto u_out = random_vec(log_size_out);
    auto u_in = random_vec(log_size_in);

    timer.start();
    generators.open(data_tensor, c, u_out, u_in);
    cout << "Current CUDA status: " << cudaGetLastError() << endl;
    timer.stop();
    cout << timer.getTotalTime() << endl;
    timer.reset();

	delete[] cpu_data;
	return 0;
}