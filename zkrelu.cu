#include "zkrelu.cuh" 

DEVICE Fr_t ulong_to_scalar(unsigned long num) {
    return {static_cast<uint32_t>(num), static_cast<uint32_t>(num >> 32), 0, 0, 0, 0, 0, 0};
}

DEVICE unsigned long scalar_to_ulong(Fr_t num){
    return static_cast<unsigned long>(num.val[0]) | (static_cast<unsigned long>(num.val[1]) << 32);
}

KERNEL void relu_kernel(Fr_t* X, Fr_t* Z, Fr_t* sign, Fr_t* mag_bin, Fr_t* rem_bin, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    Fr_t x_unmont = blstrs__scalar__Scalar_unmont(X[gid]);
    unsigned long mag;
    
    if (blstrs__scalar__Scalar_gte({4294967295U, 32767U, 0U, 0U, 0U, 0U, 0U, 0U}, x_unmont)) // positive
    {
        sign[gid] = blstrs__scalar__Scalar_ONE;
        mag = scalar_to_ulong(x_unmont); // (static_cast<unsigned long>(x_unmont.val[1]) << 32) & static_cast<unsigned long>(x_unmont.val[0]) ;
    }
    else if (blstrs__scalar__Scalar_gte(x_unmont, {1U, 4294934527U, 4294859774U, 1404937218U, 161601541U, 859428872U, 698187080U, 1944954707U})) // negative
    {
        sign[gid] = blstrs__scalar__Scalar_ZERO;
        mag = scalar_to_ulong(blstrs__scalar__Scalar_add(x_unmont, {0U, 32768U, 0U, 0U, 0U, 0U, 0U, 0U}));
    }
    bool rem_sign = mag & 32768UL;
    uint rem_mag = static_cast<uint>(mag & 32767UL);
    int rem = rem_sign ? (static_cast<int>(rem_mag) - (1 << 15)) : static_cast<int>(rem_mag); 
    uint mag_rescaled = static_cast<uint>((mag - rem) >> 16);

    #pragma unroll
    for(uint i = 0; i < 32; ++ i) mag_bin[gid * 32 + i] = ((mag_rescaled >> i) & 1) ? blstrs__scalar__Scalar_ONE: blstrs__scalar__Scalar_ZERO;

    #pragma unroll
    for(uint i = 0; i < 15; ++ i) rem_bin[gid * 16 + i] = ((rem_mag >> i) & 1) ? blstrs__scalar__Scalar_ONE: blstrs__scalar__Scalar_ZERO;
    rem_bin[gid * 16 + 15] = rem_sign ? blstrs__scalar__Scalar_ONE: blstrs__scalar__Scalar_ZERO;

    Z[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont({mag_rescaled, 0, 0, 0, 0, 0, 0, 0}), sign[gid]);
}


FrTensor zkReLU::operator()(const FrTensor& X) {
    reset_ptrs(X.size);
    FrTensor out(X.size);

    relu_kernel<<<(X.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(X.gpu_data, out.gpu_data, sign_ptr->gpu_data, mag_bin_ptr->gpu_data, rem_bin_ptr->gpu_data, X.size);
    cudaDeviceSynchronize();
    return out;
}

void zkReLU::reset_ptrs(uint size)
{
    if (sign_ptr) delete sign_ptr;
    if (mag_bin_ptr) delete mag_bin_ptr;
    if (rem_bin_ptr) delete rem_bin_ptr;

    sign_ptr = new FrTensor(size);
    mag_bin_ptr = new FrTensor(size * 32);
    rem_bin_ptr = new FrTensor(size * 16);
}

zkReLU::~zkReLU()
{
    if (sign_ptr) delete sign_ptr;
    if (mag_bin_ptr) delete mag_bin_ptr;
    if (rem_bin_ptr) delete rem_bin_ptr;
    sign_ptr = nullptr;
    mag_bin_ptr = nullptr;
    rem_bin_ptr = nullptr;
}

const uint log_Q = 5;
const uint Q = 32;
const uint log_R = 4;
const uint R = 16;

void zkReLU::prove(const FrTensor& X, const FrTensor& Z)
{
    if (X.size != Z.size) throw std::runtime_error("Incompatible dimensions");
    uint log_size = ceilLog2(X.size);

    // sumcheck for binaries
    auto u_z_bin = random_vec(log_size + log_Q);
    auto v_z_bin = random_vec(log_size + log_Q);
    auto u_r_bin = random_vec(log_size + log_R);
    auto v_r_bin = random_vec(log_size + log_R); 
    auto u_recover = random_vec(log_size);

    binary_sumcheck(*mag_bin_ptr, u_z_bin, v_z_bin);
    mag_bin_ptr -> partial_me(u_recover, Q);
    binary_sumcheck(*rem_bin_ptr, u_r_bin, v_r_bin);
    rem_bin_ptr -> partial_me(u_recover, R);

    // sumcheck for relu forward
    auto u_hp = random_vec(log_size);
    auto v_hp = random_vec(log_size);
    hadamard_product_sumcheck(X, *sign_ptr, u_hp, v_hp);
}
