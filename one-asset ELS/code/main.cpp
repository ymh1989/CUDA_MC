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

#include <algorithm>

using namespace std;

int main() {
	try {
		// Variables
		// declare variables and constants		
		const double T = 3.0;
		const size_t N_SIMULS = 10000;
		const size_t N_STEPS = 360 * (int)T; // calendar convention : 360days

		const double B = 0.6; // Knock-in barrier

		const double S0_1 = 100.0;
		const double sig1 = 0.3;

		const double r = 0.0165;
		const double dummy = 0.075;

		const double stk[] = { 0.95, 0.9, 0.85, 0.8, 0.75, 0.7};
		const double coupon[] = { 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075 };

		// this exmample set 6 observation dates.
		const double date[] = { ceil(N_STEPS * 1.0 / length), ceil(N_STEPS * 2.0 / length), 
			ceil(N_STEPS * 3.0 / length), ceil(N_STEPS * 4.0 / length),
			ceil(N_STEPS * 5.0 / length), ceil(N_STEPS * 6.0 / length) };

		bool flag = 1; // Greek flag (0 : not calcute greeks, 1 : calculate greeks)
		double diff = 0.01; // 1% diff for greeks
		///////////////////////////////////////////////

		// make variables
		const size_t N_NORMALS = N_SIMULS*N_STEPS;
		double dt = double(T) / double(N_STEPS);
		double sqrdt = sqrt(dt);

		// exception handling
		if (!(sizeof(stk) == sizeof(coupon)) && (sizeof(coupon) == sizeof(date)) && (length == (sizeof(coupon) == sizeof(date)))) {
			cout << "Size error!" << endl;
			return 0;
		}

		// generate info arrays
		const unsigned Size = sizeof(stk) / sizeof(double);
		double payment[Size] = { 0 };
		for (int i = 0; i < Size; i++) {
			payment[i] = 10000 * (1 + coupon[i]);
		}

		dev_array<double> d_stk(Size); d_stk.set(stk, Size);
		dev_array<double> d_payment(Size); d_payment.set(payment, Size);
		dev_array<double> d_date(Size); d_date.set(date, Size);

		// generate blank arrays
		double zeros[N_SIMULS] = {0};
		vector<double> s(N_SIMULS);
		dev_array<double> d_s(N_SIMULS);

		dev_array<double> d_normals(N_NORMALS);

		optionData o1(S0_1, S0_1, r, T, sig1, dt, sqrdt, B, dummy);
		optionData o2(S0_1 * (1.0 + 0.5*diff), S0_1, r, T, sig1, dt, sqrdt, B, dummy); // for greeks
		optionData o3(S0_1 * (1.0 - 0.5*diff), S0_1, r, T, sig1, dt, sqrdt, B, dummy); // for greeks

		// make a book
		optionData book[] = {o1, o2, o3};
		
		cout << "****************** INFO ******************\n";
		cout << "Strike for ELS : ";
		for (int i = 0; i < Size; i++) cout << stk[i] << " ";
		cout << endl;
		cout << "Coupon for ELS : ";
		for (int i = 0; i < Size; i++) cout << coupon[i] << " ";
		cout << endl;
		cout << "Date for ELS : ";
		for (int i = 0; i < Size; i++) cout << date[i] << " ";
		cout << endl << endl;
		cout << "Number of Paths: " << N_SIMULS << "\n";
		cout << "Number of Steps: " << N_STEPS << "\n";
		cout << "Underlying Initial Price: " << S0_1 << "\n";
		cout << "Barrier: " << B << "\n";
		cout << "Time to Maturity: " << T << " years\n";
		cout << "Risk-free Interest Rate: " << r << "\n";
		cout << "Volatility: " << sig1 << "\n";
		cout << "Face Value: " << 10000 << "\n";


		// call the kernel
		if (!flag) {
			// generate random numbers (host API)
			curandGenerator_t curandGenerator;
			curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
			curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));
			curandGenerateNormalDouble(curandGenerator, d_normals.getData(), N_NORMALS, 0.0, 1.0);
			double t1 = double(clock()) / CLOCKS_PER_SEC;

			KiELS1(book[0], d_s.getData(), d_stk.getData(), d_payment.getData(), d_date.getData(),
				d_normals.getData(), N_STEPS, N_SIMULS);	

			cudaDeviceSynchronize();	

			// copy results from device to host
			d_s.get(&s[0], N_SIMULS);
			cudaFree(d_s.getData());

			// compute the payoff average
			double gpu_sum = 0.0;
			for (size_t i = 0; i<N_SIMULS; i++) {
				gpu_sum += s[i];
			}

			gpu_sum /= N_SIMULS;
			double t2 = double(clock()) / CLOCKS_PER_SEC;


			cout << "****************** PRICE ******************\n";
			cout << "Option Price (GPU): " << gpu_sum << "\n";
			cout << "******************* TIME *****************\n";
			cout << "GPU Monte Carlo Computation: " << (t2 - t1)*1e3 << " ms\n";
			cout << "******************* END *****************\n";
		}
		else {
			double payoff[3] = {}; // save 3 payoff for obtaining price & greeks
			double t1 = double(clock()) / CLOCKS_PER_SEC;
			// generate random numbers  (host API)
			// recycle random number for greeks
			curandGenerator_t curandGenerator;
			curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
			curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));
			curandGenerateNormalDouble(curandGenerator, d_normals.getData(), N_NORMALS, 0.0, 1.0);

			for (int i = 0; i < 3; i++) {	
				KiELS1(book[i], d_s.getData(), d_stk.getData(), d_payment.getData(), d_date.getData(),
					d_normals.getData(), N_STEPS, N_SIMULS);	

				cudaDeviceSynchronize();	

				// copy results from device to host
				d_s.get(&s[0], N_SIMULS);

				// compute the payoff average
				double gpu_sum = 0.0;
				for (size_t j = 0; j < N_SIMULS; j++) {
					gpu_sum += s[j];
				}
				gpu_sum /= N_SIMULS;

				payoff[i] = gpu_sum;

			}
			double t2 = double(clock()) / CLOCKS_PER_SEC;

			double delta = (payoff[1] - payoff[2]) / (diff*S0_1); // central difference
			double gamma = (payoff[2] - 2*payoff[0] + payoff[1]) / 
				((0.5*diff*S0_1)*(0.5*diff*S0_1));
			cout << "****************** PRICE, GREEK ******************\n";
			cout << "Option Price (GPU): " << payoff[0] << "\n";
			cout << "Option Delta (GPU): " << delta << "\n";
			cout << "Option Gamma (GPU): " << gamma << "\n";
			cout << "******************* TIME *****************\n";
			cout << "GPU Monte Carlo Computation: " << (t2 - t1)*1e3 << " ms\n";
			cout << "******************* END *****************\n";

			// destroy generator
			curandDestroyGenerator(curandGenerator);
		}		
	}
	catch (exception& e) {
		cout << "exception: " << e.what() << "\n";
	}
}