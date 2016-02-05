#include <stdio.h>
#include <cstdlib>
#include <omp.h>

#include "misc.h"
#include "boundarycondition3d.h"
#include "array3d.h"

void restrict_3d(double*&  X, int& N);
void restrict_3d_full_weighting(double*& X, int& N);

/* This is the restriction method. X is used as both: an input argument, and as an output argument. 
The size of the returned X is (N/2)*(N/2)*(N/2). N is assumed to be always odd, and follows 
this rule:N = 2^k-1, for k = 1, 2, ..*/
void restrict_3d(double*& X, int& N)
{
	int new_N  =  ((N-1)/2)*((N-1)/2)*((N-1)/2);
	double *new_X = new double[new_N];
	double temp = 0;
	
	#pragma omp parallel for private(temp)
	for(int k = 1; k < N; k+=2)
	{
		for(int j = 1; j < N; j+=2)
		{
			for(int i = 1; i < N; i+=2)
			{
				temp = 0.5 * evaluate_array3d(X,N,i,j,k);
				temp += (1.0/24.0) * (evaluate_array3d(X,N,i+1,j,k) 
						    + evaluate_array3d(X,N,i-1,j,k) 
						    + evaluate_array3d(X,N,i,j+1,k) 
						    + evaluate_array3d(X,N,i,j-1,k) 
						    + evaluate_array3d(X,N,i,j,k+1) 
						    + evaluate_array3d(X,N,i,j,k-1));

				temp += (1.0/48.0) * ( evaluate_array3d(X,N,i+1,j+1,k) 
						     + evaluate_array3d(X,N,i-1,j+1,k) 
						     + evaluate_array3d(X,N,i+1,j-1,k) 
					 	     + evaluate_array3d(X,N,i-1,j-1,k) 
						     + evaluate_array3d(X,N,i+1,j,k+1) 
						     + evaluate_array3d(X,N,i-1,j,k+1) 
						     + evaluate_array3d(X,N,i,j+1,k+1) 
						     + evaluate_array3d(X,N,i,j-1,k+1) 
						     + evaluate_array3d(X,N,i+1,j,k-1) 
						     + evaluate_array3d(X,N,i-1,j,k-1) 
						     + evaluate_array3d(X,N,i,j+1,k-1) 
						     + evaluate_array3d(X,N,i,j-1,k-1));
				
				fill_array3d(new_X, (N/2), (i/2), (j/2), (k/2), temp);
			}
		}
	}
	double *old_ref = X;
	X = new_X;
	delete[] old_ref;
	N = (N-1)/2;
}


/* This is the restriction method. X is used as both: an input argument, and as an output argument. 
The size of the returned X is (N/2)*(N/2)*(N/2). N is assumed to be always odd, and follows 
this rule:N = 2^k-1, for k = 1, 2, ..
This function uses the full-weighting formula in Zhang's paper, page 452, figure (2)*/
void restrict_3d_full_weighting(double*& X, int& N)
{
	int new_N  =  ((N-1)/2)*((N-1)/2)*((N-1)/2);
	double *new_X = new double[new_N];
	double temp = 0;
	
	#pragma omp parallel for private(temp)
	for(int k = 1; k < N; k+=2)
	{
		for(int j = 1; j < N; j+=2)
		{
			for(int i = 1; i < N; i+=2)
			{
				temp = (1.0/8.0) * evaluate_array3d(X,N,i,j,k);
				temp += (1.0/16.0) * (evaluate_array3d(X,N,i+1,j,k) 
						    + evaluate_array3d(X,N,i-1,j,k) 
						    + evaluate_array3d(X,N,i,j+1,k) 
						    + evaluate_array3d(X,N,i,j-1,k) 
						    + evaluate_array3d(X,N,i,j,k+1) 
						    + evaluate_array3d(X,N,i,j,k-1));

				temp += (1.0/32.0) * ( evaluate_array3d(X,N,i-1,j-1,k)
						    +  evaluate_array3d(X,N,i+1,j-1,k)
						    +  evaluate_array3d(X,N,i-1,j+1,k)
						    +  evaluate_array3d(X,N,i+1,j+1,k)
						    +  evaluate_array3d(X,N,i,j-1,k-1)
						    +  evaluate_array3d(X,N,i,j+1,k-1)
						    +  evaluate_array3d(X,N,i,j-1,k+1)
						    +  evaluate_array3d(X,N,i,j+1,k+1)
						    +  evaluate_array3d(X,N,i-1,j,k-1)
						    +  evaluate_array3d(X,N,i+1,j,k-1)
						    +  evaluate_array3d(X,N,i-1,j,k+1)
						    +  evaluate_array3d(X,N,i+1,j,k+1));
						     

				temp += (1.0/64.0) * ( evaluate_array3d(X,N,i-1,j-1,k-1) 
						    +  evaluate_array3d(X,N,i+1,j-1,k-1)
						    +  evaluate_array3d(X,N,i-1,j-1,k+1) 
						    +  evaluate_array3d(X,N,i+1,j-1,k+1) 
						    +  evaluate_array3d(X,N,i-1,j+1,k-1) 
						    +  evaluate_array3d(X,N,i+1,j+1,k-1) 
						    +  evaluate_array3d(X,N,i-1,j+1,k+1) 
						    +  evaluate_array3d(X,N,i+1,j+1,k+1));
						     
				
				fill_array3d(new_X, (N/2), (i/2), (j/2), (k/2), temp);
			}
		}
	}
	double *old_ref = X;
	X = new_X;
	delete[] old_ref;
	N = (N-1)/2;
}


