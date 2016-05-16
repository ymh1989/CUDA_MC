#include <stdio.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include "dev_matrix.h"
#include <curand.h>
#include "chol.h"

#include <algorithm>

using namespace std;

int main() {
	try {
		// Variables
		// declare variables and constants		
		const double T = 3.0;
		const size_t N_SIMULS = 10000;
		const size_t N_STEPS = 360 * (int)T; // calendar convention : 360days
		const double B = 0.6;
		const double S0_1 = 2081.18; const double S0_2 = 3674.05;
		const double sig1 = 0.2379; const double sig2 = 0.2330;

		const double r = 0.0165;
		const double dummy = 0.075;	

		// this exmample set 6 observation dates.
		const double stk[] = { 0.95, 0.9, 0.85, 0.8, 0.75, 0.7 };
		const double coupon[] = { 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075 };
		const double date[] = { ceil(N_STEPS * 1.0 / length), ceil(N_STEPS * 2.0 / length),
			ceil(N_STEPS * 3.0 / length), ceil(N_STEPS * 4.0 / length),
			ceil(N_STEPS * 5.0 / length), ceil(N_STEPS * 6.0 / length) };

		const double rho = 0.5; // correlation between underlying 1 and 2
		double M[2 * 2] = { 0 };

		makeChol2(M, rho); // cholesky decomposition for correlated random number

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
		vector<double> s(N_SIMULS);
		dev_array<double> d_s(N_SIMULS);
		dev_matrix<double> d_normals(N_NORMALS, 2); // initial random number in GPU
		dev_matrix<double> chol(2, 2);

		dev_matrix<double> d_normals_rev(N_NORMALS, 2); // correlated random number in GPU
		chol.set(M, 2, 2);

		// generate random numbers (host API)
		dev_fillRand(d_normals.getData(), N_NORMALS, 2);

		// make a correlated random number (GPU matrix multiplication using cublas)
		dev_mmul(d_normals.getData(), chol.getData(), d_normals_rev.getData(), N_NORMALS, 2, 2);

		d_normals.~dev_matrix(); chol.~dev_matrix(); // destruct unnecessary array for memory space

		optionData o1(S0_1, S0_1, r, T, sig1, dt, sqrdt, B, dummy); // zero tick
		optionData o2(S0_2, S0_2, r, T, sig2, dt, sqrdt, B, dummy);
		optionData o3(S0_1 * (1.0 + 0.5*diff), S0_1, r, T, sig1, dt, sqrdt, B, dummy); // up tick
		optionData o4(S0_2 * (1.0 + 0.5*diff), S0_2, r, T, sig2, dt, sqrdt, B, dummy);
		optionData o5(S0_1 * (1.0 - 0.5*diff), S0_1, r, T, sig1, dt, sqrdt, B, dummy); // down tick
		optionData o6(S0_2 * (1.0 - 0.5*diff), S0_2, r, T, sig2, dt, sqrdt, B, dummy);

		// make a book
		optionData book[] = { o1, o2, o3, o4, o5, o6 };

		double payoff[5] = {};
		double gpu_sum;
		double t1 = double(clock()) / CLOCKS_PER_SEC;
		for (int i = 0; i < 5; i++) {
			// call the kernel
			if (i == 0) { // s1:0 s2:0
				ELS2(book[0], book[1], d_s.getData(), d_stk.getData(), d_payment.getData(), d_date.getData(), d_normals_rev.getData(), N_STEPS, N_SIMULS);
			}
			else if (i == 1) // s1:0 s2:+
			{
				ELS2(book[0], book[3], d_s.getData(), d_stk.getData(), d_payment.getData(), d_date.getData(), d_normals_rev.getData(), N_STEPS, N_SIMULS);
			}
			else if (i == 2) // s1:0 s2:-
			{
				ELS2(book[0], book[5], d_s.getData(), d_stk.getData(), d_payment.getData(), d_date.getData(), d_normals_rev.getData(), N_STEPS, N_SIMULS);
			}
			else if (i == 3) // s1:+ s2:0
			{
				ELS2(book[1], book[2], d_s.getData(), d_stk.getData(), d_payment.getData(), d_date.getData(), d_normals_rev.getData(), N_STEPS, N_SIMULS);
			}
			else if (i == 4) // s1:- s2:0
			{
				ELS2(book[1], book[4], d_s.getData(), d_stk.getData(), d_payment.getData(), d_date.getData(), d_normals_rev.getData(), N_STEPS, N_SIMULS);
			}

			cudaDeviceSynchronize();

			// copy results from device to host
			d_s.get(&s[0], N_SIMULS);

			// compute the payoff average
			gpu_sum = 0.0;
			for (size_t j = 0; j < N_SIMULS; j++) {
				gpu_sum += s[j];
			}

			gpu_sum /= N_SIMULS;
			payoff[i] = gpu_sum;
		}
		double t2 = double(clock()) / CLOCKS_PER_SEC;

		double delta1 = (payoff[1] - payoff[2]) / (diff*S0_1);
		double delta2 = (payoff[4] - payoff[3]) / (diff*S0_2);
		double gamma1 = (payoff[1] - 2 * payoff[0] + payoff[2]) /
			((0.5*diff*S0_1)*(0.5*diff*S0_1));
		double gamma2 = (payoff[4] - 2 * payoff[0] + payoff[3]) /
			((0.5*diff*S0_2)*(0.5*diff*S0_2));

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
		cout << "Underlying Initial Price: " << S0_1 << " " << S0_2 << "\n";
		cout << "Barrier: " << B << "\n";
		cout << "Time to Maturity: " << T << " years\n";
		cout << "Risk-free Interest Rate: " << r << "\n";
		cout << "Volatility: " << sig1 << " " << sig2 << "\n";
		cout << "Face Value: " << 10000 << "\n";
		cout << "****************** PRICE, GREEK ******************\n";
		cout << "Option Price (GPU): " << gpu_sum << "\n";
		cout << "Option Delta1 (GPU): " << delta1 << "\n";
		cout << "Option Gamma1 (GPU): " << gamma1 << "\n";
		cout << "Option Delta2 (GPU): " << delta2 << "\n";
		cout << "Option Gamma2 (GPU): " << gamma2 << "\n";
		cout << "******************* TIME *****************\n";
		cout << "GPU Monte Carlo Computation: " << (t2 - t1)*1e3 << " ms\n";
		cout << "******************* END *****************\n";
	}
	catch (exception& e) {
		cout << "exception: " << e.what() << "\n";
	}
}