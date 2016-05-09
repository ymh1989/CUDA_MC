#ifndef _CHOL_H_
#define _CHOL_H_

// cholesky decomposition
void chol3(double dest[], double src[][3])
{
	int i = 0, j = 0, k = 0, n = 3;
	double tmparr[3][3] = { 0 };

	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum3 = 0.0;

	tmparr[0][0] = sqrt(src[0][0]);
	for (j = 1; j <= n - 1; j++)
		tmparr[j][0] = src[j][0] / tmparr[0][0];
	for (i = 1; i <= (n - 2); i++)
	{
		for (k = 0; k <= (i - 1); k++)
			sum1 += pow(tmparr[i][k], 2);
		tmparr[i][i] = sqrt(src[i][i] - sum1);
		for (j = (i + 1); j <= (n - 1); j++)
		{
			for (k = 0; k <= (i - 1); k++)
				sum2 += tmparr[j][k] * tmparr[i][k];
			tmparr[j][i] = (src[j][i] - sum2) / tmparr[i][i];
		}
	}
	for (k = 0; k <= (n - 2); k++)
		sum3 += pow(tmparr[n - 1][k], 2);
	tmparr[n - 1][n - 1] = sqrt(src[n - 1][n - 1] - sum3);

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			dest[i * 3 + j] = tmparr[i][j];
		}
	}
}

void makeChol3(double dest[], const double * src)
{
	int i, j;
	unsigned cnt = 0;
	double tmp_corr[3][3] = { 0 };
	for (i = 0; i < 3; i++){
		for (j = 0; j <= i; j++) {
			if (i == j) {
				tmp_corr[j][i] = 1.0;
			}
			else {
				tmp_corr[j][i] = src[cnt++];
				tmp_corr[i][j] = tmp_corr[j][i];
			}
		}
	}

	chol3(dest, tmp_corr);
}

#endif