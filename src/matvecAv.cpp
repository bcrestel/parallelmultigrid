/* matvecAv7p returns the matrix-vector product A.v, where
v is the solution vector for a 3d problem stored in a 1d-vector
as defined in array3d.cpp, and A is the 3D laplacian discretized
with finite-differences and a 7-point stencil.
N is the number of interior grid points along one dimension.
*/

#include <omp.h>

#include "array3d.h"


double* matvecAv7p( const double* const array3d, const int N )
{
	double* output = new double[N*N*N];

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
				value = ( 6. * evaluate_array3d( array3d, N, ii, jj, kk ) - ( ngh1 + ngh2 + ngh3 + ngh4 + ngh5 + ngh6 )) / (h*h);
				fill_array3d( output, N, ii, jj, kk, value );}
		}
	}

	return output;
}

