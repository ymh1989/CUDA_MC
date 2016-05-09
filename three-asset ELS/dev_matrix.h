#ifndef _DEV_MATRIX_H_
#define _DEV_MATRIX_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

// This header is referred at follow link.
// https://www.quantstart.com/articles/Monte-Carlo-Simulations-In-CUDA-Barrier-Option-Pricing
// modified by Minhyun Yoo
// Easy form for CUDA syntax

template <typename T>
class dev_matrix
{
	// public functions
public:
	explicit dev_matrix()
		: start_(0),
		end_(0)
	{}

	// constructor
	explicit dev_matrix(size_t row, size_t col)
	{
		allocate(row*col);
	}
	// destructor
	~dev_matrix()
	{
		free();
	}

	// resize the vector
	void resize(size_t row, size_t col)
	{
		free();
		allocate(row*col);
	}

	// get the size of the array
	size_t getSize() const
	{
		return end_ - start_;
	}

	// get data
	const T* getData() const
	{
		return start_;
	}

	T* getData()
	{
		return start_;
	}

	// set
	void set(const T* src, size_t row, size_t col)
	{
		size_t min = std::min(row*col, getSize());
		cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("failed to copy to device memory");
		}
	}
	// get
	void get(T* dest, size_t row, size_t col)
	{
		size_t min = std::min(row*col, getSize());
		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("failed to copy to host memory");
		}
	}


	// private functions
private:
	// allocate memory on the device
	void allocate(size_t size)
	{
		cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
		if (result != cudaSuccess)
		{
			start_ = end_ = 0;
			throw std::runtime_error("failed to allocate device memory");
		}
		end_ = start_ + size;
	}

	// free memory on the device
	void free()
	{
		if (start_ != 0)
		{
			cudaFree(start_);
			start_ = end_ = 0;
		}
	}

	T* start_;
	T* end_;
};

#endif