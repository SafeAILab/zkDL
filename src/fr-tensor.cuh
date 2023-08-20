#ifndef FR_TENSOR_CUH
#define FR_TENSOR_CUH

#include "bls12-381.cuh"

typedef blstrs__scalar__Scalar Fr_t;

class FrTensor
{   
    private:
    const uint size;
    Fr_t* gpu_data;

    public:
    FrTensor(uint size): size(size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
    }

    FrTensor(uint size, const Fr_t* cpu_data): size(size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
        cudaMemcpy(gpu_data, cpu_data, sizeof(Fr_t) * size, cudaMemcpyHostToDevice);
    }

    FrTensor(const FrTensor& t): size(t.size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
        cudaMemcpy(gpu_data, t.gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToDevice);
    }

    Fr_t operator()(uint idx)
    {
        Fr_t out;
        cudaMemcpy(&out, gpu_data + idx, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        return out;
    }

    FrTensor operator+(const FrTensor &) const;

    FrTensor operator+(const Fr_t &) const;

    FrTensor& operator+=(const FrTensor &);
    
    FrTensor& operator+=(const Fr_t &);

    FrTensor operator-() const;

    FrTensor operator-(const FrTensor &) const;

    FrTensor operator-(const Fr_t &) const;

    FrTensor& operator-=(const FrTensor &);
    
    FrTensor& operator-=(const Fr_t &);

    ~FrTensor()
    {
        cudaFree(gpu_data);
        gpu_data = nullptr;
    }

};

#endif