/* prolongation3d: compute the solution for a 3d problem
on a finer grid. The solution is provided in the form of
a 1d-vector, as defined in array3d.cpp, and the size N
is taken as the number of grid points along one dimension.
*/

#include <cstdlib>

#include <omp.h>

#include "array3d.h"
#include "misc.h"

void prolongation3d( double*& array3d, int& N )
{
	int N_new = 2*N + 1;
	int sizearray_new = N_new*N_new*N_new;
	double* array3d_new = new double[sizearray_new];

	int uniqueid;
	double value, value1, value2, value3, value4;

	int ii, jj, kk;

	#pragma omp parallel for private(ii, jj, uniqueid, value, value1, value2, value3, value4)
	for (kk = 0; kk < N_new; kk++){
		for (jj = 0; jj < N_new; jj++){
			for (ii = 0; ii < N_new; ii++){
				uniqueid = index2unique( ii, jj, kk );

				switch (uniqueid)
				{
					case 0: // ii, jj, kk odd
						value = evaluate_array3d( array3d, N, (ii-1)/2, (jj-1)/2, (kk-1)/2 );
						fill_array3d( array3d_new, N_new, ii, jj, kk, value );
						break;
	
					case 1: // ii even,	jj, kk odd
						value1 = evaluate_array3d( array3d, N, ii/2 - 1, (jj-1)/2, (kk-1)/2 );
						value2 = evaluate_array3d( array3d, N, ii/2, (jj-1)/2, (kk-1)/2 );
						value = (value1 + value2) / 2.;
						fill_array3d( array3d_new, N_new, ii, jj, kk, value );
						break;

					case 2: // jj even, 	ii, kk odd
						value1 = evaluate_array3d( array3d, N, (ii-1)/2, jj/2 - 1, (kk-1)/2 );
						value2 = evaluate_array3d( array3d, N, (ii-1)/2, jj/2, (kk-1)/2 );
						value = (value1 + value2) / 2.;
						fill_array3d( array3d_new, N_new, ii, jj, kk, value );
						break;

					case 3: // ii, jj even,	kk odd
						value1 = evaluate_array3d( array3d, N, ii/2 - 1, jj/2 - 1, (kk-1)/2 );
						value2 = evaluate_array3d( array3d, N, ii/2, jj/2 - 1, (kk-1)/2 );
						value3 = evaluate_array3d( array3d, N, ii/2 - 1, jj/2, (kk-1)/2 );
						value4 = evaluate_array3d( array3d, N, ii/2, jj/2, (kk-1)/2 );
						value = (value1 + value2 + value3 + value4) / 4.;
						fill_array3d( array3d_new, N_new, ii, jj, kk, value );
						break;

					case 4: // kk even, 	ii, jj odd
						value1 = evaluate_array3d( array3d, N, (ii-1)/2, (jj-1)/2, kk/2 - 1 );
						value2 = evaluate_array3d( array3d, N, (ii-1)/2, (jj-1)/2, kk/2 );
						value = (value1 + value2) / 2.;
						fill_array3d( array3d_new, N_new, ii, jj, kk, value );
						break;

					case 5: // ii, kk even,	jj odd
						value1 = evaluate_array3d( array3d, N, ii/2 - 1, (jj-1)/2, kk/2 );
						value2 = evaluate_array3d( array3d, N, ii/2, (jj-1)/2, kk/2 );
						value3 = evaluate_array3d( array3d, N, ii/2 - 1, (jj-1)/2, kk/2 - 1 );
						value4 = evaluate_array3d( array3d, N, ii/2, (jj-1)/2, kk/2 -1 );
						value = (value1 + value2 + value3 + value4) / 4.;
						fill_array3d( array3d_new, N_new, ii, jj, kk, value );
						break;

					case 6: // jj, kk even,	ii odd
						value1 = evaluate_array3d( array3d, N, (ii-1)/2, jj/2 - 1, kk/2 );
						value2 = evaluate_array3d( array3d, N, (ii-1)/2, jj/2, kk/2 );
						value3 = evaluate_array3d( array3d, N, (ii-1)/2, jj/2 - 1, kk/2 - 1 );
						value4 = evaluate_array3d( array3d, N, (ii-1)/2, jj/2, kk/2 - 1 );
						value = (value1 + value2 + value3 + value4) / 4.;
						fill_array3d( array3d_new, N_new, ii, jj, kk, value );
						break;

					case 7: // ii, jj, kk even
						value1 = evaluate_array3d( array3d, N, ii/2, jj/2 - 1, kk/2 );
						value2 = evaluate_array3d( array3d, N, ii/2, jj/2, kk/2 );
						value3 = evaluate_array3d( array3d, N, ii/2, jj/2 - 1, kk/2 - 1 );
						value4 = evaluate_array3d( array3d, N, ii/2, jj/2, kk/2 - 1 );
						value = (value1 + value2 + value3 + value4) / 8.;
						value1 = evaluate_array3d( array3d, N, ii/2 - 1, jj/2 - 1, kk/2 );
						value2 = evaluate_array3d( array3d, N, ii/2 - 1, jj/2, kk/2 );
						value3 = evaluate_array3d( array3d, N, ii/2 - 1, jj/2 - 1, kk/2 - 1 );
						value4 = evaluate_array3d( array3d, N, ii/2 - 1, jj/2, kk/2 - 1 );
						value += (value1 + value2 + value3 + value4) / 8.;
						fill_array3d( array3d_new, N_new, ii, jj, kk, value );
						break;

				}
			}
		}
	}
	// Update arguments:
	N = N_new;
	delete[] array3d;
	array3d = array3d_new;
}
