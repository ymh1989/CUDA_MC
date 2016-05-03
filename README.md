##Monte Calro simulation using CUDA##

###Introduction###
This repo contains an implementation of pricing financial derivatives using Monte Calo simulation with CUDA(Compute Unified Device Architecture).

###Environment###
- GPU : NVIDIA GeForce GTX 650 @ 1.072GHZ GDDR5 1GB
- [CUDA toolkit 7.5](https://developer.nvidia.com/cuda-toolkit)
- CPU : Intel(R) Core(TM) i5-6400 @ 2.7GHZ 
- RAM : DDR3L 16GB PC3-12800
- Microsoft Visual Studio Community 2013

###Result###
- In this repo, I compare the performance between CPU and GPU. The parameters can be modified freely.

 Case | European call
------------ | -------------
GPU | 88ms
CPU | 275ms
** As you can see in `Environment`, the GPU which I tested is old type(2012 late), however, the CPU is latest model(2016 early). So please understand that there is no marked difference in computational cost.


###Note###
- You need to add `curand.lib` files as linker input in the development environment.
- Also, the platform you are targetting in VS configuration manager should be `x64`, since `curand.lib` is `x64` library.
- If you're interested in my works, please visit my [homepage](https://sites.google.com/site/yoomh1989/).