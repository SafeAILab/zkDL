#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"
#include <iostream>
#include <iomanip>
#include "timer.hpp"

using namespace std;

int main(int argc, char *argv[])
{
	// uint size = stoi(argv[1]);
	// uint window_size = stoi(argv[2]);
	// bool need_print = false;
	// if (argc > 3) need_print = stoi(argv[3]);

	uint log_size = stoi(argv[1]);
	uint size = 1 << log_size;

	Fr_t* cpu_data = new Fr_t[size];
	for (uint i = 0; i < size; ++ i)
	{
		cpu_data[i] = {i, 2 * i + 1, 3 * i + 2, 4 * i + 3, 5 * i + 4, 6 * i + 5, 7 * i + 6, 8 * i + 7};
	}

	cout << "size=" << size << endl;
	//cout << "window_size=" << size << endl;
	Timer timer;

	FrTensor t(size, cpu_data);

	vector<Fr_t> u1 (log_size);
	for (uint i = 0; i < u1.size(); ++ i) u1[i] = {0, 0, 0, 0, 0, 0, 0, 0};

	timer.start();
	auto y1 = t(u1);
	timer.stop();
	cout << timer.getTotalTime() << endl;
	cout << y1 << endl;
	timer.reset();

	srand(time(NULL));
	vector<Fr_t> u2 (log_size);
	uint rand_idx = rand() % size;
	cout << "Testing at random index "<< rand_idx << endl; 
	for (uint i = 0; i < u2.size(); ++ i)
	{
		if ((rand_idx >> i) & 1U) u2[i] = { 4294967294, 1, 215042, 1485092858, 3971764213, 2576109551, 2898593135, 405057881 };
		else u2[i] = {0, 0, 0, 0, 0, 0, 0, 0};
	} 

	timer.start();
	auto y2 = t(u2);
	timer.stop();
	cout << timer.getTotalTime() << endl;
	cout << y2 << endl;


	

	// FrTensor t(size, cpu_data);
	// timer.start();
	// auto tp = t.split(window_size);
	// timer.stop();
	// cout << timer.getTotalTime() << endl;
	// timer.reset();

	// auto& t0 = tp.first;
	// auto& t1 = tp.second;

	// timer.start();
	// auto tp0 = t0.split(window_size);
	// timer.stop();
	// cout << timer.getTotalTime() << endl;
	// timer.reset();

	// cout << t0.size << "\t" << t1.size << endl;

	// if (need_print)
	// {
	// 	for (uint i = 0; i < t0.size; ++ i) cout << t0(i) << endl;
	// 	for (uint i = 0; i < t1.size; ++ i) cout << t1(i) << endl;
	// }
	
	
	cout << "Current CUDA status: " << cudaGetLastError() << endl;

	delete[] cpu_data;
	return 0;
}