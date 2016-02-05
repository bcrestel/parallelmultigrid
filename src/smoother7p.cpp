/* smoother7p: applies relaxation to a 1d-vector representing
a 3d-array, as defined in array3d.cpp, and using a
7-point stencil in finite-differences.
*/

#include <omp.h>

#include "array3d.h"

void smoother7p( double*& array3d, const int N, double (*fluxfct) (const int&, const int&, const int&, const int&), double omega )
{
	double*	array3d_new = new double[N*N*N];

	double h = 1. / (N + 1.);

	int ii, jj, kk;
	double value, ngh1, ngh2, ngh3, ngh4, ngh5, ngh6;

	#pragma omp parallel for private(ii, jj, value, ngh1, ngh2, ngh3, ngh4, ngh5, ngh6)
	for (kk = 0; kk < N; kk++){
		for (jj = 0; jj < N; jj++){
			for (ii = 0; ii < N; ii++){
				ngh1 = evaluate_array3d( array3d, N, ii-1, jj, kk );
				ngh2 = evaluate_array3d( array3d, N, ii+1, jj, kk );
				ngh3 = evaluate_array3d( array3d, N, ii, jj-1, kk );
				ngh4 = evaluate_array3d( array3d, N, ii, jj+1, kk );
				ngh5 = evaluate_array3d( array3d, N, ii, jj, kk-1 );
				ngh6 = evaluate_array3d( array3d, N, ii, jj, kk+1 );
				value = ( h*h*fluxfct( N, ii, jj, kk ) + ngh1 + ngh2 + ngh3 + ngh4 + ngh5 + ngh6 ) / 6.;
				fill_array3d( array3d_new, N, ii, jj, kk, (1.-omega)*evaluate_array3d( array3d, N, ii, jj, kk ) + omega*value );}
		}
	}

	delete[] array3d;
	array3d = array3d_new;
}

void smoother7p_3dfun( double*& array3d, const int N, double* fluxfct, double omega )
{
	double*	array3d_new = new double[N*N*N];

	double h = 1. / (N + 1.);

	int ii, jj, kk;
	double value, ngh1, ngh2, ngh3, ngh4, ngh5, ngh6;

	#pragma omp parallel for private(ii, jj, value, ngh1, ngh2, ngh3, ngh4, ngh5, ngh6)
	for (kk = 0; kk < N; kk++){
		for (jj = 0; jj < N; jj++){
			for (ii = 0; ii < N; ii++){
				ngh1 = evaluate_array3d( array3d, N, ii-1, jj, kk );
				ngh2 = evaluate_array3d( array3d, N, ii+1, jj, kk );
				ngh3 = evaluate_array3d( array3d, N, ii, jj-1, kk );
				ngh4 = evaluate_array3d( array3d, N, ii, jj+1, kk );
				ngh5 = evaluate_array3d( array3d, N, ii, jj, kk-1 );
				ngh6 = evaluate_array3d( array3d, N, ii, jj, kk+1 );
				value = ( h*h* evaluate_array3d(fluxfct, N, ii, jj, kk ) + ngh1 + ngh2 + ngh3 + ngh4 + ngh5 + ngh6 ) / 6.;
				fill_array3d( array3d_new, N, ii, jj, kk, (1.-omega)*evaluate_array3d( array3d, N, ii, jj, kk ) + omega*value );}
		}
	}

	delete[] array3d;
	array3d = array3d_new;
}
