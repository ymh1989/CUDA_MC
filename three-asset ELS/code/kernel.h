#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_
#define length 6 // # of observation
 
#define MIN_USERDEFINE(X, Y) (((X) < (Y)) ? (X) : (Y))
#define CEIL(a, b) (((a)+(b)-1) / (b))
typedef struct optionData
{
	double S0;
	double S0_ref;
	double r;
	double discr;
	double T;
	double sigma;
	double dt;
	double sqrdt;

	double B;
	double dummy;


	optionData(double _S0,
		double _S0_ref,
		double _r,
		double _discr,
		double _T,
		double _sigma,
		double _dt,
		double _sqrdt,
		double _B,
		double _dummy)
	{
		S0 = _S0; S0_ref = _S0_ref; r = _r;
		discr = _discr; T = _T;
		sigma = _sigma; dt = _dt;
		sqrdt = _sqrdt;
		B = _B; dummy = _dummy;
	}

}optionData;

void ELS3(optionData option1, optionData option2, optionData option3, double * d_s, double * stk, double * payment, double * date, double * d_normals, unsigned N_STEPS, unsigned N_SIMULS);

void dev_fillRand(double *A, size_t rows_A, size_t cols_A);

void dev_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n);
#endif