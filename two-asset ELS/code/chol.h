#ifndef _CHOL_H_
#define _CHOL_H_

// cholesky decomposition
void makeChol2(double dest[], const double src)
{
	//chol2(dest, tmp_corr);
	dest[0] = 1.0; dest[1] = 0.0;
	dest[2] = src; dest[3] = sqrt(1.0 - src*src);
}

#endif