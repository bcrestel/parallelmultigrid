#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include <omp.h>

#include "array3d.h"
#include "boundarycondition3d.h"
#include "matvecAv.h"
#include "misc.h"
#include "prolongation3d.h"
#include "restriction3d.h"
#include "smoother7p.h"
#include "timer.h"


/*This function implemented one v-cycle*/
void v_cycle(double*& array3d, double h, int N, double* function_3d, int v1, int v2);

int main(int argc, char **argv)
{
	int k = (argc > 1) ? atoi(argv[1]) : 5;
	int N = pow(2, k) - 1;  // N should follow this rule N = 2^k-1
	std::cout << "Solve Poisson problem with MG for " << N*N*N << " grid points." << std::endl;

	int num_threads;
	#pragma omp parallel
	{
		#pragma omp master
		num_threads = omp_get_num_threads();
	}

	// create a 3D array
	double* array = new double[N*N*N];
	// populate the array with random values:
	int i, j;
	double val;
	#pragma omp parallel for private(i, j, val)
	for(int k = 0; k < N; k++){
		for(j = 0; j < N; j++){
			for(i = 0; i < N; i++){
				val = (double)random()/RAND_MAX;
				fill_array3d(array, N, i, j, k, val);}
		}
	}
	

	// create a 3D array - fluxfct ( f = 0 )
	double* fluxfct = new double[N*N*N];
	// populate the array with zeros:
	#pragma omp parallel for private(i, j)
	for(int k = 0; k < N; k++){
		for(int j = 0; j < N; j++){
			for(int i = 0; i < N; i++){
				fill_array3d(fluxfct, N, i, j, k, 0);}
		}
	}



	//printf("\nOriginal Array:\n");
	//print_double_array(array, N);

	double h = 1. / (N + 1.);
	int v1 = 5;
	int v2 = 5;
	double norm_2;
	int iter = 10;
	double t1, t2, tt = 0.0;

	for (int ii = 0; ii < iter; ii++){
		t1 = gtod_timer();
		v_cycle(array, h, N, fluxfct, v1, v2);
		t2 = gtod_timer();
		tt += t2 - t1;

		//norm_2 = norm2_omp(array, N*N*N);
		//printf("v-cycle #%d, 2-norm = %1f\n", ii+1, norm_2);
	}
	
	std::cout << "N = " << N << ", "
		  << "Nb threads = " << num_threads << ".\n"
		  << "After " << iter << " iterations, 2-norm = " 
		  << norm2_omp(array, N*N*N) << ", inf-norm = " << norminf_omp(array, N*N*N) << ". "
		  << "Running time = " << tt << " msec.\n" << std::endl;

	//printf("\nAfter one v-cycle:\n");
	//print_double_array(array, N);

	delete[] array;
	delete[] fluxfct;
	
        return 0;
}


void v_cycle(double*& array3d, double h, int N, double* function_3d, int v1, int v2)
{
	// base case
	if(N == 1){
		smoother7p_3dfun(array3d, N, function_3d );}
	else{
		// (1) pre-smooth (v1) times:
		for(int i = 0; i < v1; i++)
			smoother7p_3dfun(array3d, N, function_3d );

		// (2) compute residual vector:
		double* mat_vec = matvecAv7p(array3d, N);
		double *residual = vecSub_fun3d(mat_vec, function_3d, N);
		delete[] mat_vec;

		// (3) restrict residual:
		restrict_3d_full_weighting(residual, N);  // N will become floor(N/2)

		// (4) make the initial guess for subsequent calls = 0 (as we are solving for the error !)
		double*	e_2h = new double[N*N*N];
		#pragma omp parallel for
		for(int i = 0; i < (N*N*N); i++){
			e_2h[i] = 0;}

		// (5) recurse: 
		v_cycle(e_2h, 2*h, N, residual, v1, v2);
		delete[] residual;

		// (6) prolongate:
		prolongation3d(e_2h, N); // N will become N = 2*N+1
		double* addition =  vecAdd(array3d, e_2h, N);
		delete[] array3d;
		array3d = addition;

		// (7) post-smooth (v2) times ..
		for(int i = 0; i < v2; i++)
			smoother7p_3dfun(array3d, N, function_3d );
	}
}


