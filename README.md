##Monte Calro simulation using CUDA##

###Introduction###
This repo contains an implementation of pricing financial derivatives using Monte Calo simulation with CUDA(Compute Unified Device Architecture).

###Environment###
- GPU : NVIDIA GeForce GTX 650 1GB
- [CUDA toolkit 7.5](https://developer.nvidia.com/cuda-toolkit)
- CPU : Intel(R) Core(TM) i5-6400 @ 2.7GHZ 
- RAM : DDR3L 16GB PC3-12800
- Microsoft Visual Studio Community 2013

###Result###
- In this repo, I compare the performance between CPU and GPU. The parameter can be modified freely.

First Header | Second Header
------------ | -------------
Content cell 1 | Content cell 2
Content column 1 | Content column 2

###Note###
- You need to add `curand.lib` files as linker input in the development environment.
- Also, the platform you are targetting in VS configuration manager should be `x64`, since `curand.lib` is `x64` library.
- As you can see in `Environment` 