#include "kernel.h"

// up-and-out call
__global__ void Kernel_up_out_barrier_single(
	optionData data,
	double * d_s,
	double * d_normals,
	unsigned N_SIMULS,
	unsigned N_SIMULS)
{
	int s_idx = threadIdx.x + blockIdx.x * blockDim.x; // thread index
	int n_idx = (s_idx)* N_SIMULS; // for random number indexing

	// check thread # < # of simuls
	if (s_idx < N_SIMULS) {
		int n = 0;

		double s_curr = data.S0;
		double T = data.T;
		double sig = data.sig;
		double r = data.r;
		double dt = data.dt;
		double sqrdt = data.sqrdt;
		double K = data.K;
		double B = data.B;

		double payoff = 0.0;
		bool tag = 0; // tag for path-dependent property
		do {
			s_curr = s_curr * exp((r - (sig*sig)*0.5)*dt + sig*sqrdt*d_normals[n_idx]);
			
			tag = (s_curr > B) ? 1 : tag; // check knock-out (if s > B, tag = 1, otherwise tag is retained.)

			n_idx++; // random number index
			n++; // time stepping
		} while (n < N_SIMULS);

		// payoff using ternary operator
		payoff = tag ? 0 : ((s_curr > K) ? (s_curr - K) : 0);

		// to save results, sycronize threads
		__syncthreads();

		// save payoff
		d_s[s_idx] = payoff;
	}
}

void up_out_barrier_single(
	optionData option,
	double * d_s,
	double * d_normals,
	unsigned N_SIMULS,
	unsigned N_SIMULS) {
	const unsigned BLOCK_SIZE = 1024; // # of threads in a block (1-dimension threads & block)
	const unsigned GRID_SIZE = CEIL(N_SIMULS, BLOCK_SIZE); // # of block in a grid
	Kernel_up_out_barrier_single << <GRID_SIZE, BLOCK_SIZE >> >
		(option, d_s, d_normals, N_SIMULS, N_SIMULS);
}