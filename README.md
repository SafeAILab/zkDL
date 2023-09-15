# zkDL: Deep Learning with Zero-knowledge Proofs on CUDA

**zkDL** is a specialized backend that combines zero-knowledge proofs (ZKP) with deep learning, specifically optimized for CUDA.

---

## Introduction

**zkDL** represents a significant step in integrating zero-knowledge proofs with deep learning. It uniquely emphasizes the preservation of tensor structures and harnesses the parallel processing power of CUDA, resulting in efficient proof computations.

## Technical Overview

- **Foundation**: This project is based on the CUDA implementation of the `bls12-381` elliptic curve, using the `ec-gpu` package developed by Filecoin.
  
- **Quantization**: For the efficient application of ZKP tools, the floating point numbers involved in deep learning computations are quantized.
    
- **Tensor Structures and GKR Protocol**: We utilize a specialized version of the GKR protocol to maintain tensor structures, facilitating the parallelization of proofs. For operations like ReLU, which are inherently non-arithmetic and thus challenging for ZKP schemes, *auxiliary inputs* are employed to transition them into arithmetic operations.

- **Neural Network Modelling**: We approach neural network modeling by visualizing it as an arithmetic circuit. Our strategy breaks free from the conventional layer-wise precedence, especially when non-arithmetic operations come into play, allowing for a more efficient 'flattened' circuit representation.

![Circuit Diagram](./images/circuit.png)

## Prerequisites

Ensure CUDA is installed on your system, and find out the compatible CUDA architecture. Let's assume that we have `sm_70` architecture, which is common for Nvidia Tesla V100.

## Setup & Installation

1. Set the architecture using NVCC_FLAGS in the `Makefile`:

```cmake
# NVCC compiler flags
NVCC_FLAGS := -arch=sm_70
```

2. Compile the demonstration:

```bash
make demo
```

Note that it is normal that the compilation takes a long time (up to a few minutes). We are working on a fix.

## Running the Demo

To initiate the demo:

```bash
# ./demo num_layer log_width log_batch_size
./demo 8 10 6 
```
This command will run an inference on a fully connected neural network with 8 layers, 1024 neurons per layer, and a batch size of 64. The entire process, including initialization, should conclude in a few seconds.

## Performance Metrics

Benchmarking, based on [ModulusLabs' criteria](https://medium.com/@ModulusLabs/chapter-5-the-cost-of-intelligence-da26dbf93307), has shown zkDL's capability to achieve a speed-up in proof time by factors ranging from 100x to 1000x.

![Benchmark Graph](./images/benchmark.png)

## Future Development

- Enhance support for various deep learning architectures.
- Re-introduce zero-knowledge verifiable **training** alongside **inference** as detailed in [zkDL: Efficient Zero-Knowledge Proofs of Deep Learning Training](https://arxiv.org/abs/2307.16273).
- Focus on proof compression across deep learning layers and explore a multi-gpu version for enhanced performance.
- Broaden the range of structures and back propagations to increase adaptability.

## Disclaimer

Please note that zkDL is currently in the development stage and has not undergone thorough auditing. Production use is not recommended at this time.

## Contact & Contribution

For those interested in contributing to zkDL or seeking further information, please reach out to the project lead, Haochen Sun, at haochen.sun@uwaterloo.ca.

---

I hope this version strikes the balance you're looking for.