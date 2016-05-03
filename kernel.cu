#include "kernel.h"

__global__ void Kernel_Vanilla_Call_single(
	optionData data,
	double * d_s,
	double * d_normals,
	unsigned N_STEPS,
	unsigned N_PATHS)
{
	int s_idx = threadIdx.x + blockIdx.x * blockDim.x; // i
	int n_idx = (s_idx)* N_STEPS;

	if (s_idx < N_PATHS) {
		int n = 0;

		double s_curr = data.S0;
		double T = data.T;
		double sig = data.sig;
		double r = data.r;
		double dt = data.dt;
		double sqrdt = data.sqrdt;
		double K = data.K;

		double payoff = 0.0;

		do {
			s_curr = s_curr * exp((r - (sig*sig)*0.5)*dt + sig*sqrdt*d_normals[n_idx]);
			
			n_idx++;
			n++;
		} while (n < N_STEPS);

		payoff = (s_curr > K)  ? (s_curr - K) : 0;

		__syncthreads();

		d_s[s_idx] = payoff;
	}
}

void Vanilla_Call_single(
	optionData option,
	double * d_s,
	double * d_normals,
	unsigned N_STEPS,
	unsigned N_PATHS) {
	const unsigned BLOCK_SIZE = 1024;
	const unsigned GRID_SIZE = CEIL(N_PATHS, BLOCK_SIZE);
	Kernel_Vanilla_Call_single << <GRID_SIZE, BLOCK_SIZE >> >
		(option, d_s, d_normals, N_STEPS, N_PATHS);
}