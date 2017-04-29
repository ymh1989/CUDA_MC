##  Monte Carlo simulation to option pricing in CUDA

### Introduction
This repo contains an implementation of pricing financial derivatives using Monte Calo simulation with CUDA(Compute Unified Device Architecture).

### Environment
- GPU : NVIDIA GeForce GTX 650 @ 1.072GHZ GDDR5 1GB
- [CUDA toolkit 7.5](https://developer.nvidia.com/cuda-toolkit)
- CPU : Intel(R) Core(TM) i5-6400 @ 2.7GHZ 
- RAM : DDR3L 16GB PC3-12800
- Microsoft Visual Studio Community 2013

### Result
- In this repo, I compare the performance between CPU and GPU. The parameters can be modified freely.
| European call | UP&out call | ELS 1 asset<p>(price&greeks)| ELS 2 asset<p>(price&greeks) | ELS 3 asset<p>(price&greeks)
------------ | ------------- | ------------- | ------------- | -------------
GPU | 88ms <p>(10<sup>7</sup> simuls)</p> | 251ms <p>(10<sup>5</sup> simuls)</p>| 129ms <p>(10<sup>4</sup> simuls)</p> | 223ms <p>(10<sup>4</sup> simuls) | 833ms <p>(10<sup>4</sup> simuls)
CPU | 275ms <p>(10<sup>7</sup> simuls)</p> | 484ms <p>(10<sup>5</sup> simuls)</p>| N/A | N/A | N/A
** As you can see in `Environment`, the GPU which I tested is old type(2012 late), however, the CPU is latest model(2016 early). So please understand that there is no marked difference in computational cost.


### Note
- You need to add `curand.lib` files as linker input in the development environment.
- Also, the platform you are targetting in VS configuration manager should be `x64`, since `curand.lib` is `x64` library.
- If you're interested in my works, please [email](mailto:yoomh1989@gmail.com) me.
