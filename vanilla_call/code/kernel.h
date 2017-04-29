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

	// constructor
	optionData(double _S0,
		double _r,
		double _T,
		double _sig,
		double _dt,
		double _sqrdt,
		double _K = 0)
	{
		S0 = _S0; r = _r; T = _T;
		sig = _sig; dt = _dt;
		sqrdt = _sqrdt; K = _K;
	}

}optionData;

void Vanilla_Call_single(optionData option, double * d_s, double * d_normals, unsigned N_SIMULS, unsigned N_SIMULS);

#endif