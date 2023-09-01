#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "proof.cuh"
#include <iostream>
#include <iomanip>
#include "timer.hpp"

using namespace std;

// const uint CommitNumGroups = 80;
// const uint CommitWindowSizes = 5;
// const uint CommitNumWindows = (256 + CommitWindowSizes - 1) / CommitWindowSizes;

int main(int argc, char *argv[])
{
	// uint size = stoi(argv[1]);
	// uint window_size = stoi(argv[2]);
	// bool need_print = false;
	// if (argc > 3) need_print = stoi(argv[3]);

	
	uint size = stoi(argv[1]);
    G1TensorJacobian gt(size, G1Jacobian_generator);

	Fr_t* cpu_data = new Fr_t[size];
	for (uint i = 0; i < size; ++ i)
	{
		cpu_data[i] = {i, 0, 0, 0, 0, 0, 0, 0};
	}

    FrTensor t(size, cpu_data);

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

    timer.start();
    auto gt3 = gt1 + gt2;
    timer.stop();
    cout << timer.getTotalTime() << endl;
    timer.reset();

    timer.start();
    auto gt4 = gt + gt;
    timer.stop();
    cout << timer.getTotalTime() << endl;
    timer.reset();

    for (uint i = 0; i < 3; ++ i)
    {   
        cout << "At index " << i << ":" << endl;
        cout << gt1(i) << endl;
        cout << gt2(i) << endl;
        cout << gt3(i) << endl;
        cout << gt4(i) << endl;
    }
	
	
	cout << "Current CUDA status: " << cudaGetLastError() << endl;

	delete[] cpu_data;
	return 0;
}