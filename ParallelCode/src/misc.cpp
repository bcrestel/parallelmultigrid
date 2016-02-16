/* Misc. routines.
*/

#include <cstdlib>

#include <omp.h>

#include "misc.h"

double norm2_omp( const double* const V, const int Ntotal )
{
	double output = 0.;

	#pragma omp parallel for reduction(+:output)
	for (int ii = 0; ii < Ntotal; ii++){
		output += V[ii]*V[ii];}

	return sqrt( output );
}


double norminf_omp( const double* const V, const int Ntotal )
{
	double output = 0.;
	double tmp = 0.;

	#pragma omp parallel firstprivate(tmp) shared(output)
	{
		#pragma omp for
		for (int ii = 0; ii < Ntotal; ii++){
			tmp = (tmp < fabs(V[ii])) ? fabs(V[ii]) : tmp;}

		#pragma omp critical
		output = (tmp > output) ? tmp : output;
	}

	return output;
}


int index2unique( const int ii, const int jj, const int kk )
{
	int iiev = Is_even(ii) ? 1 : 0;
	int jjev = Is_even(jj) ? 1 : 0;
	int kkev = Is_even(kk) ? 1 : 0;

	return (iiev + 2*jjev + 2*2*kkev);
}
