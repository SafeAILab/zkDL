#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "proof.cuh"
#include <iostream>
#include <iomanip>
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

	
	uint size = stoi(argv[1]);
    uint m = stoi(argv[2]);
    Commitment generators(size, G1Jacobian_generator);

    

	Fr_t* cpu_data = new Fr_t[m * size];
	for (uint i = 0; i < size; ++ i)
	{
		cpu_data[i] = {i, 0, 0, 0, 0, 0, 0, 0};
	}

    FrTensor t(m * size, cpu_data);

    generators *= t;

	cout << "size=" << size << endl;
	//cout << "window_size=" << size << endl;
	Timer timer;
    
    timer.start();
    auto gt1 = gt * t;
    timer.stop();
    cout << timer.getTotalTime() << endl;
    timer.reset();

    timer.start();
    auto gt2 = gt * -t;
    timer.stop();
    cout << timer.getTotalTime() << endl;
    timer.reset();
	
	
	cout << "Current CUDA status: " << cudaGetLastError() << endl;

	delete[] cpu_data;
	return 0;
}