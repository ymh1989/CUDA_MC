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
		// Variables
		// declare variables and constants		
		const size_t N_PATHS = 1000000;
		const size_t N_STEPS = 100;
		
		const double S0_1 = 100;
		const double T = 1.0;
		const double K = 100;
		const double sig1 = 0.3;
		const double r = 0.03;

		int n_idx = 0;
		double payoff = 0.0;

		// make variables
		const size_t N_NORMALS = N_PATHS*N_STEPS;
		double dt = double(T) / double(N_STEPS);
		double sqrdt = sqrt(dt);
		///////////////////////////////////////////////
		
		
		// generate blank arrays
		vector<double> s(N_PATHS);
		dev_array<double> d_s(N_PATHS);
		dev_array<double> d_normals(N_NORMALS);

		optionData o1(S0_1, r, T, sig1, dt, sqrdt, K);
		
		// generate random numbers (host API)
		curandGenerator_t curandGenerator;
		curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
		curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));
		curandGenerateNormalDouble(curandGenerator, d_normals.getData(), N_NORMALS, 0.0, 1.0);

		// GPU start 
		double t1 = double(clock()) / CLOCKS_PER_SEC;	

		// call the kernel
		Vanilla_Call_single(o1, d_s.getData(), d_normals.getData(), N_STEPS, N_PATHS);

		cudaDeviceSynchronize();	

		// copy results from device to host
		d_s.get(&s[0], N_PATHS);

		// compute the payoff average
		double gpu_sum = 0.0;
		for (size_t i = 0; i<N_PATHS; i++) {
			gpu_sum += s[i];
		}
		gpu_sum /= N_PATHS;
		gpu_sum *= exp(-r*T);
		double t2 = double(clock()) / CLOCKS_PER_SEC;

		// CPU start
		vector<double> normals(N_NORMALS);
		d_normals.get(&normals[0], N_NORMALS);
		double cpu_sum = 0.0;
		double s_curr = 0.0;
		for (size_t i = 0; i < N_PATHS; i++) {
			n_idx = i*N_STEPS;

			s_curr = S0_1;

			int n = 0;

			do {
				s_curr = s_curr * exp((r - (sig1*sig1)*0.5)*dt + sig1*sqrdt*normals[n_idx]);
				n_idx++;
				n++;
			} while (n < N_STEPS);

			payoff = (s_curr > K ? s_curr - K : 0.0);
			cpu_sum += payoff;
		}

		cpu_sum /= exp(-r*T) * N_PATHS;

		double t3 = double(clock()) / CLOCKS_PER_SEC;

		cout << "****************** INFO ******************\n";
		cout << "S0 : ";
		cout << S0_1 << endl;
		cout << "Strike : ";
		cout << K << endl;
		cout << "Maturity : ";
		cout << T << " year(s)" << endl;
		cout << "Volatility : ";
		cout << sig1 << endl;
		cout << "Risk-free Interest Rate : ";
		cout << r << endl;
		cout << "Number of Simulations: " << N_PATHS << "\n";
		cout << "Number of Steps: " << N_STEPS << "\n";

		cout << "****************** PRICE ******************\n";
		cout << "Option Price (GPU): " << gpu_sum << "\n";
		cout << "Option Price (CPU): " << cpu_sum << "\n";
		cout << "******************* TIME *****************\n";
		cout << "GPU Monte Carlo Computation: " << (t2 - t1)*1e3 << " ms\n";
		cout << "CPU Monte Carlo Computation: " << (t3 - t2)*1e3 << " ms\n";
		cout << "******************* END *****************\n";

		// destroy generator
		curandDestroyGenerator(curandGenerator);
		}	
	catch (exception& e) {
		cout << "exception: " << e.what() << "\n";
	}
}