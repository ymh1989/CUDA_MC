#include "kernel.h"

__global__ void KiELS1_kernel(
	optionData data,
	double * d_s,
	double * stk, 
	double * payment, 
	double * date,
	double * d_normals,
	unsigned N_STEPS,
	unsigned N_SIMULS
	)
{
	int s_idx = threadIdx.x + blockIdx.x * blockDim.x; // thread index
	int n_idx = (s_idx)* N_STEPS; // for random number indexing

	if (s_idx < N_SIMULS) {
		// Initialize
		double s_curr = data.S0;
		double s_ref = data.S0_ref;
		double sigma = data.sigma;
		double r = data.r;
		double dt = data.dt;
		double sqrdt = data.sqrdt;
		double B = data.B;
		double dummy = data.dummy;
		double s_curr_cal = -1.0;

		double drift = (r - (sigma*sigma)*0.5)*dt;
		double sigsqdt = sigma*sqrdt;

		double payoff = 0.0;
		unsigned int cnt1 = 0;
		unsigned int cnt2 = 0;
		double idx[length] = { 0 };
		int n = 0;

		bool tag = 0;
		bool kievent = 0;
		s_curr_cal = s_curr / s_ref;
		do {
			// Geometric Brownian motion
			s_curr_cal = s_curr_cal * exp(drift + sigsqdt*d_normals[n_idx]);

			// cheeck knock-in event
			kievent = (s_curr_cal < B) ? 1 : kievent;

			// save underlying price at observation dates
			if ((n+1) == date[cnt1]) {
				idx[cnt1] = s_curr_cal;
				cnt1++;
			}

			n_idx++; // random number index
			n++; // time stepping
		} while (n < N_STEPS);

		// check observation dates (early redemption)
		for (int i = 0; i < length; i++) {
			if (idx[i] >= stk[i]) {
				payoff = payment[i];
				tag = 1;
				cnt2 = i;
				break;
			}
		}
		if (tag == 0) {
			// payoff using ternary operator
			payoff = 10000 * s_curr_cal;	
			payoff = (kievent == 0) ? ((s_curr_cal >= B) ? 10000 * (1 + dummy) : payoff) : payoff;
			cnt2 = length - 1;
		}
		// payoff using ternary operator (calendar convention : 360days)
		payoff = payoff * exp(-r * date[cnt2] / 360.0);

		// to save results, sycronize threads
		__syncthreads();

		// save payoff
		d_s[s_idx] = payoff;
	}
}

void KiELS1(
	optionData option,
	double * d_s,
	double * stk,
	double * payment,
	double * date,
	double * d_normals,
	unsigned N_STEPS,
	unsigned N_SIMULS
	) {
		const unsigned BLOCK_SIZE = 1024; // # of threads in a block (1-dimension threads & block)
		const unsigned GRID_SIZE = CEIL(N_SIMULS, BLOCK_SIZE); // # of block in a grid
		KiELS1_kernel << <GRID_SIZE, BLOCK_SIZE >> >(
			option, d_s, stk, payment, date, d_normals, N_STEPS, N_SIMULS);
}