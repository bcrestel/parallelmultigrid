#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "array3d.h"
#include "prolongation3d.h"
#include "misc.h"
#include "smoother7p.h"

void testarray3d()
{
	const int N = 10;
	double* array3d = new double[N*N*N];

	for (int ii = 0; ii < N; ii++){
		for (int jj = 0; jj < N; jj++){
			for (int kk = 0; kk < N; kk++){
				fill_array3d( array3d, N, ii, jj, kk, ii+jj+kk );}
		}
	}

	for (int ii = 0; ii < N; ii++){
		for (int jj = 0; jj < N; jj++){
			for (int kk = 0; kk < N; kk++){
				if (fabs( evaluate_array3d( array3d, N, ii, jj, kk )) - (ii+jj+kk) > 1e-3){
					printf("evaluate_array3d(%d,%d,%d) = %f.\n", ii, jj, kk, evaluate_array3d( array3d, N, ii, jj, kk ));}
			}
		}
	}
}


void test_prolongation()
{
	int N = 3;
	double* array3d = new double[N*N*N];

	for (int ii=0; ii < N*N*N; ii++){
		array3d[ii] = 0.;}
	printf("\n\nPrint before prolongation.\n");
	print_allslicesf( array3d, N );

	printf("\n\nProlongation.\n");
	prolongation3d( array3d, N );

	printf("\n\nPrint after prolongation.\n");
	print_allslicesf( array3d, N );
}


void test_misc()
{
/*	int N = 3;
	double* array3d = new double[N*N*N];

	for (int ii=0; ii < N*N*N; ii++){
		array3d[ii] = 1;}
	print_allslices( array3d, N );*/

	for (int ii = 0; ii < 2; ii++){
		for (int jj = 0; jj < 2; jj++){
			for (int kk = 0; kk < 2; kk++){
				printf("index2unique(%d,%d,%d) = %d.\n", ii, jj, kk, index2unique(ii, jj, kk));}
		}
	}
}


double fluxfunction( const int& N, const int& ii, const int& jj, const int& kk ){	return 0.;}

void test_smoother7p()
{
	int N = 3;
	double* array3d = new double[N*N*N];

	for (int ii = 0; ii < N*N*N; ii++){	array3d[ii] = 1.;}
	printf("\n\nPrint before smoothing.\n");
	print_allslicesf( array3d, N );

	printf("\n\nSmoothing.\n");
	smoother7p( array3d, N, fluxfunction );

	printf("\n\nPrint after smoothing.\n");
	print_allslicesf( array3d, N );
}


int main()
{
//	testarray3d();
//	test_prolongation();
//	test_misc();
	test_smoother7p();

	return 0;
}
