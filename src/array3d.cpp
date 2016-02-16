/* array3d.cpp: Misc. functions to manipulate a vector
representing a 3d array, i.e an array with 3 indices.

The 3d array is a single vector, i.e a pointer to double.
The size is understood as the number of grid points along one direction. 
Therefore, the total size of the vector is N*N*N, where N is the
number of grid points along one direction. Note also that we only store
the interior grid points.
*/

#include <cstdio>
#include <iostream>

#include "misc.h"
#include "boundarycondition3d.h"


void fill_array3d( double* const array, const int N, 
		const int i, const int j, const int k, const double value )
{
	if ( Is_inbetween( i, 0, N-1 ) && Is_inbetween( j, 0, N-1 ) && Is_inbetween( k, 0, N-1 ) ){
		array[ i + j*N + k*N*N ]  = value;}
	else{
		printf("Error in fill_array3d(): index out of range.\n");
	}
}


double evaluate_array3d( const double* const array, const int N, 
			const int i, const int j, const int k )
{
	if ( Is_inbetween( i, 0, N-1 ) && Is_inbetween( j, 0, N-1 ) && Is_inbetween( k, 0, N-1 ) ){
		return ( array[ i + j*N + k*N*N ] );
	}
	else if ( (i==-1) || (i==N) || (j==-1) || (j==N) || (k==-1) || (k==N) ){
		return boundarycondition3d( N, i, j, k );
	}
	else{
		printf("Error in evaluate_array3d(): index out of range.\n");}
}


void print_slice( const double* const array3d, const int N, 
		const int direction, const int index )
{
	switch (direction)
	{
		case 0:
			for (int jj = 0; jj < N; jj++){
				for (int kk = 0;  kk < N; kk++){
					printf("array3d[%d][%d][%d] = %f.\n", index, jj, kk, 
					evaluate_array3d( array3d, N, index, jj, kk ));}}
			break;

		case 1:
			for (int ii = 0; ii < N; ii++){
				for (int kk = 0;  kk < N; kk++){
					printf("array3d[%d][%d][%d] = %f.\n", ii, index, kk, 
					evaluate_array3d( array3d, N, ii, index, kk ));}}
			break;

		case 2:
			for (int ii = 0; ii < N; ii++){
				for (int jj = 0;  jj < N; jj++){
					printf("array3d[%d][%d][%d] = %f.\n", ii, jj, index,  
					evaluate_array3d( array3d, N, ii, jj, index ));}}
			break;
	}
}


void print_slicef( const double* const array3d, const int N, 
		const int direction, const int index )
{
	switch (direction)
	{
		case 0:
			for (int kk = N-1;  kk >= 0; kk--){
				for (int jj = 0; jj < N; jj++){
					std::cout << evaluate_array3d( array3d, N, index, jj, kk ) << " ";}
				std::cout << std::endl;
			}
			break;

		case 1:
			for (int kk = N-1;  kk >= 0; kk--){
				for (int ii = 0; ii < N; ii++){
					std::cout << evaluate_array3d( array3d, N, ii, index, kk ) << " ";}
				std::cout << std::endl;
			}
			break;

		case 2:
			for (int jj = N-1;  jj >= 0; jj--){
				for (int ii = 0; ii < N; ii++){
					std::cout << evaluate_array3d( array3d, N, ii, jj, index ) << " ";}
				std::cout << std::endl;
			}
			break;
	}
}


void print_allslices( const double* const array3d, const int N )
{
	for (int ii = 0; ii < N; ii++){
		printf( "\nPrint slice %d in direction 0.\n", ii );
		print_slice( array3d, N, 0, ii );}
}


void print_allslicesf( const double* const array3d, const int N )
{
	for (int ii = 0; ii < N; ii++){
		printf( "\nPrint slice %d in direction 0.\n", ii );
		print_slicef( array3d, N, 0, ii );}
}


/*This funciton is used to subtract an "array3d" vector from a "3d" function (output = fluxfct - array_3d_1) .. */
double* vecSub(const double* const array3d_1, double (*fluxfct) (const int, const int, const int, const int), const int N)
{
	double* output = new double[N*N*N];
	double result;

	#pragma omp parallel for private(result)
	for(int kk = 0; kk < N; kk++)
	{
		for(int jj = 0; jj < N; jj++)
		{
			for(int ii = 0; ii < N; ii++)
			{
				result = fluxfct( N, ii, jj, kk) - evaluate_array3d( array3d_1, N, ii, jj, kk) ;
				fill_array3d( output, N, ii, jj, kk, result);
			}
		}
	}
	return output;
}


/*This funciton is used to subtract an "array3d" vector from a "3d" function (output = fluxfct - array_3d_1) .. */
double* vecSub_fun3d(const double* const array3d_1, double* fluxfct, const int N)
{
	double* output = new double[N*N*N];
	double result;
	int ii, jj;

	#pragma omp parallel for private(result, ii, jj)
	for(int kk = 0; kk < N; kk++){
		for(jj = 0; jj < N; jj++){
			for(ii = 0; ii < N; ii++){
				result = evaluate_array3d( fluxfct, N, ii, jj, kk ) - evaluate_array3d( array3d_1, N, ii, jj, kk );
				fill_array3d( output, N, ii, jj, kk, result );}
		}
	}
	return output;
}


/*This function returns the addtion of two "array3d" vectors*/
double* vecAdd(const double* const array3d_1, const double* const array3d_2, const int N)
{
	double* output = new double[N*N*N];
	double result;

	#pragma omp parallel for private(result)
	for(int kk = 0; kk < N; kk++)
	{
		for(int jj = 0; jj < N; jj++)
		{
			for(int ii = 0; ii < N; ii++)
			{
				result = evaluate_array3d( array3d_1, N, ii, jj, kk) + evaluate_array3d( array3d_2, N, ii, jj, kk) ;
				fill_array3d( output, N, ii, jj, kk, result);
			}
		}
	}
	return output;
}
