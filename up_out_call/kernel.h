#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_
#define CEIL(a, b) (((a)+(b)-1) / (b))

typedef struct optionData
{
	double S0;
	double r;
	double T;
	double sig;
	double dt;
	double sqrdt;

	double K;
	double B;

	// constructor
	optionData(double _S0,
		double _r,
		double _T,
		double _sig,
		double _dt,
		double _sqrdt,
		double _K = 0,
		double _B = 0)
	{
		S0 = _S0; r = _r; T = _T;
		sig = _sig; dt = _dt;
		sqrdt = _sqrdt; K = _K;
		B = _B;
	}

}optionData;

void up_out_barrier_single(optionData option, double * d_s, double * d_normals, unsigned N_STEPS, unsigned N_SIMULS);

#endif