#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"
#include <iostream>
#include <iomanip>
#include "timer.hpp"

using namespace std;

int main(int argc, char *argv[])
{
	uint size = stoi(argv[1]);
	uint window_size = stoi(argv[2]);
	bool need_print = false;
	if (argc > 3) need_print = stoi(argv[3]);

	Fr_t* cpu_data = new Fr_t[size];
	for (uint i = 0; i < size; ++ i)
	{
		cpu_data[i].val[0] = i;
	}

	cout << "size=" << size << endl;
	cout << "window_size=" << size << endl;
	Timer timer;

	FrTensor t(size, cpu_data);
	timer.start();
	auto tp = t.split(window_size);
	timer.stop();
	cout << timer.getTotalTime() << endl;
	timer.reset();

	auto& t0 = tp.first;
	auto& t1 = tp.second;

	timer.start();
	auto tp0 = t0.split(window_size);
	timer.stop();
	cout << timer.getTotalTime() << endl;
	timer.reset();

	cout << t0.size << "\t" << t1.size << endl;

	if (need_print)
	{
		for (uint i = 0; i < t0.size; ++ i) cout << t0(i) << endl;
		for (uint i = 0; i < t1.size; ++ i) cout << t1(i) << endl;
	}
	
	
	cout << "Current CUDA status: " << cudaGetLastError() << endl;

	delete[] cpu_data;
	return 0;
}