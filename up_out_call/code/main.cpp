#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include <curand.h>

using namespace std;

int main() {
	try {
		// declare variables and constants		
		const size_t N_SIMULS = 100000;
		const size_t N_SIMULS = 360;
		
		const double S0_1 = 100;
		const double T = 1.0;
		const double K = 100;
		const double sig1 = 0.3;
		const double r = 0.03;
		const double B = 130;

		// make variables
		const size_t N_NORMALS = N_SIMULS*N_SIMULS;
		double dt = double(T) / double(N_SIMULS);
		double sqrdt = sqrt(dt);
		///////////////////////////////////////////////	
		
		// generate blank arrays
		vector<double> s(N_SIMULS);
		dev_array<double> d_s(N_SIMULS);
		dev_array<double> d_normals(N_NORMALS);

		// For calculating many derivatives
		optionData o1(S0_1, r, T, sig1, dt, sqrdt, K, B);
		
		// generate random numbers (host API)
		curandGenerator_t curandGenerator;
		curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
		curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));
		curandGenerateNormalDouble(curandGenerator, d_normals.getData(), N_NORMALS, 0.0, 1.0);

		// GPU start 
		double t1 = double(clock()) / CLOCKS_PER_SEC;	

		// call the kernel
		up_out_barrier_single(o1, d_s.getData(), d_normals.getData(), N_SIMULS, N_SIMULS);

		cudaDeviceSynchronize();	

		// copy results from device to host
		d_s.get(&s[0], N_SIMULS);

		// compute the payoff average
		double gpu_sum = 0.0;
		for (size_t i = 0; i<N_SIMULS; i++) {
			gpu_sum += s[i];
		}
		gpu_sum /= N_SIMULS;
		gpu_sum *= exp(-r*T);
		double t2 = double(clock()) / CLOCKS_PER_SEC;

		// CPU start
		vector<double> normals(N_NORMALS);

		// Get random number from device to host
		d_normals.get(&normals[0], N_NORMALS);

		double cpu_sum = 0.0;
		double s_curr = 0.0;
		double payoff = 0.0;
		bool tag = 0;
		int n_idx = 0;
		int n = 0;
		for (size_t i = 0; i < N_SIMULS; i++) {
			n_idx = i*N_SIMULS;

			s_curr = S0_1;
			tag = 0;
			n = 0;

			do {
				s_curr = s_curr * exp((r - (sig1*sig1)*0.5)*dt + sig1*sqrdt*normals[n_idx]);

				tag = (s_curr > B) ? 1 : tag; // check knock-out

				n_idx++;
				n++;
			} while (n < N_SIMULS);

			payoff = tag ? 0 : ((s_curr > K) ? (s_curr - K) : 0);
			cpu_sum += exp(-r*T) * payoff;
		}

		cpu_sum /= N_SIMULS;

		double t3 = double(clock()) / CLOCKS_PER_SEC;

		cout << "************ KNOCK OUT (UP & OUT) CALL INFO ************\n";
		cout << "S0 : ";
		cout << S0_1 << endl;
		cout << "Strike : ";
		cout << K << endl;
		cout << "Barrier : ";
		cout << B << endl;
		cout << "Maturity : ";
		cout << T << " year(s)" << endl;
		cout << "Volatility : ";
		cout << sig1 << endl;
		cout << "Risk-free Interest Rate : ";
		cout << r << endl;
		cout << "Number of Simulations: " << N_SIMULS << "\n";
		cout << "Number of Steps: " << N_SIMULS << "\n";

		cout << "****************** PRICE ******************\n";
		cout << "Option Price (GPU): " << gpu_sum << "\n";
		cout << "Option Price (CPU): " << cpu_sum << "\n";
		cout << "******************* TIME *****************\n";
		cout << "GPU Monte Carlo Computation: " << (t2 - t1)*1e3 << " ms\n";
		cout << "CPU Monte Carlo Computation: " << (t3 - t2)*1e3 << " ms\n";
		cout << "******************* END *****************\n";

		// destroy random number generator
		curandDestroyGenerator(curandGenerator);
		}	
	catch (exception& e) {
		cout << "exception: " << e.what() << "\n";
	}
}