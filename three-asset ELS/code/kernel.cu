#include "kernel.h"
#include <curand.h>
#include <time.h>
#include <cublas_v2.h>

__global__ void KiELS3_kernel(
	optionData data1,
	optionData data2,
	optionData data3,
	double * d_s,
	double * stk, 
	double * payment, 
	double * date,
	double * d_normals,
	unsigned N_STEPS,
	unsigned N_SIMULS)
{
	int s_idx = threadIdx.x + blockIdx.x * blockDim.x; // thread index
	int n_idx = (s_idx) * N_STEPS; // for random number indexing	

	if (s_idx < N_SIMULS) {
		double s_curr1 = data1.S0; double sigma1 = data1.sigma; double r1 = data1.r; double dt1 = data1.dt; double sqrdt1 = data1.sqrdt; double B1 = data1.B; double dummy1 = data1.dummy;
		double s_curr2 = data2.S0; double sigma2 = data2.sigma; double r2 = data2.r; double dt2 = data2.dt; double sqrdt2 = data2.sqrdt; double B2 = data2.B; double dummy2 = data2.dummy;
		double s_curr3 = data3.S0; double sigma3 = data3.sigma; double r3 = data3.r; double dt3 = data3.dt; double sqrdt3 = data3.sqrdt; double B3 = data3.B; double dummy3 = data3.dummy;

		double ref_s1 = data1.S0_ref; double ref_s2 = data2.S0_ref; double ref_s3 = data3.S0_ref;

		double s_curr_cal1 = -1.0, s_curr_cal2 = -1.0, s_curr_cal3 = -1.0;
		double s_curr_min = -1.0;
		double payoff = 0.0;
		unsigned int cnt1 = 0;
		unsigned int cnt2 = 0;
		double idx[length] = { 0 };

		double drift1 = (r1 - (sigma1*sigma1)*0.5)*dt1, sigsqdt1 = sigma1*sqrdt1;
		double drift2 = (r2 - (sigma2*sigma2)*0.5)*dt2, sigsqdt2 = sigma2*sqrdt2;
		double drift3 = (r3 - (sigma3*sigma3)*0.5)*dt3, sigsqdt3 = sigma3*sqrdt3;

		int n = 0;
		bool tag = 0;
		bool kievent = 0;

		s_curr_cal1 = s_curr1 / ref_s1;
		s_curr_cal2 = s_curr2 / ref_s2;
		s_curr_cal3 = s_curr3 / ref_s3;

		do {
			// Geometric Brownian motion
			s_curr_cal1 = s_curr_cal1 * exp(drift1 + sigsqdt1*d_normals[n_idx]);
			s_curr_cal2 = s_curr_cal2 * exp(drift2 + sigsqdt2*d_normals[N_STEPS * N_SIMULS + n_idx]);
			s_curr_cal3 = s_curr_cal3 * exp(drift3 + sigsqdt3*d_normals[2 * N_STEPS * N_SIMULS + n_idx]);
			
			// worst performer
			s_curr_min = MIN_USERDEFINE(s_curr_cal1, s_curr_cal2);
			s_curr_min = MIN_USERDEFINE(s_curr_min, s_curr_cal3);

			// cheeck knock-in event
			kievent = (s_curr_min < B1) ? 1 : kievent;

			// save underlying price at observation dates
			if ((n+1) == date[cnt1]) {
				idx[cnt1] = s_curr_min;
				cnt1++;
			}

			n_idx++;  // random number index
			n++;  // time stepping
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
			payoff = 10000 * s_curr_min;
			payoff = (kievent == 0) ? ((s_curr_min >= B1) ? 10000 * (1 + dummy1) : payoff) : payoff;
			cnt2 = length - 1;
		}
		
		payoff = payoff * exp(-data1.discr * date[cnt2] / 360.0);

		__syncthreads();

		d_s[s_idx] = payoff;
	}
}

void ELS3(
	optionData option1,
	optionData option2,
	optionData option3,
	double * d_s,
	double * stk,
	double * payment,
	double * date,
	double * d_normals,
	unsigned N_STEPS,
	unsigned N_SIMULS) {
	const unsigned BLOCK_SIZE = 1024; // # of threads in a block (1-dimension threads & block)
	const unsigned GRID_SIZE = CEIL(N_SIMULS, BLOCK_SIZE); // # of block in a grid
	KiELS3_kernel << <GRID_SIZE, BLOCK_SIZE >> >(
		option1, option2, option3, d_s, stk, payment, date, d_normals, N_STEPS, N_SIMULS);
}

void dev_fillRand(double *A, size_t rows_A, size_t cols_A) 
{
	// random number generation host API
	curandGenerator_t rnd;
	curandCreateGenerator(&rnd, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(rnd, (unsigned long long)time(NULL));
	curandGenerateNormalDouble(rnd, A, rows_A*cols_A, 0.0, 1.0);
}

void dev_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
	// GPU matrix multiplication
    int lda = m, ldb = k, ldc = m;
    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Destroy the handle
    cublasDestroy(handle);
}