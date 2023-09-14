# zkDL: A Specialized Zero-knowledge Proof Backend for Deep Learning on CUDA

**zkDL** provides a specialized zero-knowledge proof backend tailored for deep learning on CUDA. Designed to preserve tensor structures unlike generic ZKP frameworks, zkDL optimizes proof generation by exploiting parallel computation on CUDA.

---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Performance](#performance)
- [Future Development](#future-development)
- [Disclaimer](#disclaimer)
- [Contribute](#contribute)
- [License](#license)

---

## Introduction

**zkDL** is an evolution in the integration of zero-knowledge proofs with deep learning. While most ZKP architectures cater to a general-purpose application in machine learning, zkDL uniquely upholds tensor structures. Thanks to CUDA's parallel processing capabilities, it greatly accelerates proof computation.

The project builds upon the CUDA implementation of the `bls12-381` elliptic curve, specifically the `ec-gpu` package by Filecoin. From this foundation, we've developed specialized proof protocols for various deep learning architectures.

## Prerequisites

Ensure you have CUDA installed. Please also verify the CUDA architecture compatibility (for example: sm_70).

## Setup & Installation

1. Begin by setting the desired architecture NVCC_FLAGS in the `Makefile`.

```cmake
# NVCC compiler flags
NVCC_FLAGS := -arch=sm_70
```

2. Compile the demo using the following command. It's normal for the compilation to take a few minutes.

```bash
make demo
```

## Usage

Run the demo with the command below:

```bash
# ./demo num_layer log_width log_batch_size
./demo 8 10 6 
```
The provided example runs inference on a fully connected neural network with 8 layers, 1024 neurons per layer, and a batch size of 64. The complete execution, initialization included, should finish in just a few seconds.

## Performance

*Details to be added by the author.*

## Future Development

- Enhance support for diverse deep learning architectures.
- Re-implement zero-knowledge verifiable **training**, not just **inference**, as discussed in the research paper [zkDL: Efficient Zero-Knowledge Proofs of Deep Learning Training](https://arxiv.org/abs/2307.16273).

## Disclaimer

This project is still in its development phase and has not been audited. It is not recommended for production use yet.

## Contribute

Interested in contributing to zkDL? Please see our contribution guidelines (Link if available).

## License

*License details to be added by the author.*