/* array3d.cpp: Misc. functions to manipulate a vector
representing a 3d array, i.e an array with 3 indices.

The 3d array is a single vector, i.e a pointer to double.
The size is understood as the number of grid points along one direction
+ the surrounding ghost values (int pts or bndy pts). 
Therefore, the total size of the vector is (N+2)*(N+2)*(N+2), where N is the
number of grid points along one direction. Note also that we only store
the interior grid points.

LIST OF MPI_Abort codes:
-99: fill_array3d()
-88: evaluate_array3d()
-77: evaluate_array3d_bndy()
-66: initialize_array3d()
*/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <time.h>

#include <omp.h>

#include "array3d.h"
#include "boundarycondition3d.h"


double* initialize_array3d( const int& N, double* recvarray, const int& mpirank,
		const int* const opposite_rank, MPI_Datatype* datatype_faces )
{
	int sizearray3d = (N+2)*(N+2)*(N+2);
	double* array3d = new double[sizearray3d];

	// Initialize random number generator
	srand( mpirank*time(NULL) );

	int chunk_size = 1000;
	
	#pragma omp parallel
	{
		#pragma omp for
		for (int ii = 0; ii < sizearray3d; ii++){	array3d[ii] = (rand() % 100) / 99.;}

		// Fill boundary points on faces where needed
		for (int face = 0; face < 6; face++)
		{
			if (opposite_rank[face] == MPI_PROC_NULL)
			{
				switch( face+1 )
				{
					case 1:
						#pragma omp for schedule(dynamic, chunk_size) nowait
						for (int ind1 = 0; ind1 < N+2; ind1++){
							for (int ind2 = 0; ind2 < N+2; ind2++){
								fill_array3d( array3d, N, ind2, 0, ind1, boundarycondition3d( N, 0, 0, 0 ) );}
						}
						break;
					case 2:
						#pragma omp for schedule(dynamic, chunk_size) nowait
						for (int ind1 = 0; ind1 < N+2; ind1++){
							for (int ind2 = 0; ind2 < N+2; ind2++){
								fill_array3d( array3d, N, N+1, ind2, ind1, boundarycondition3d( N, 0, 0, 0 ) );}
						}
						break;
					case 3:
						#pragma omp for schedule(dynamic, chunk_size) nowait
						for (int ind1 = 0; ind1 < N+2; ind1++){
							for (int ind2 = 0; ind2 < N+2; ind2++){
								fill_array3d( array3d, N, ind2, N+1, ind1, boundarycondition3d( N, 0, 0, 0 ) );}
						}
						break;
					case 4:
						#pragma omp for schedule(dynamic, chunk_size) nowait
						for (int ind1 = 0; ind1 < N+2; ind1++){
							for (int ind2 = 0; ind2 < N+2; ind2++){
								fill_array3d( array3d, N, 0, ind2, ind1, boundarycondition3d( N, 0, 0, 0 ) );}
						}
						break;
					case 5:
						#pragma omp for schedule(dynamic, chunk_size) nowait
						for (int ind1 = 0; ind1 < N+2; ind1++){
							for (int ind2 = 0; ind2 < N+2; ind2++){
								fill_array3d( array3d, N, ind2, ind1, N+1, boundarycondition3d( N, 0, 0, 0 ) );}
						}
						break;
					case 6:
						#pragma omp for schedule(dynamic, chunk_size) nowait
						for (int ind1 = 0; ind1 < N+2; ind1++){
							for (int ind2 = 0; ind2 < N+2; ind2++){
								fill_array3d( array3d, N, ind2, ind1, 0, boundarycondition3d( N, 0, 0, 0 ) );}
						}
						break;
					default:
						MPI_Abort( MPI_COMM_WORLD, -66 );
				}
			}
		}
	}

	MPI_Request	my_requests[2];

	/* Read boundary points from priority directions, i.e
	faces 4, 1, 6. */
	int faces2check[3]; faces2check[0] = 1; faces2check[1] = 4; faces2check[2] = 6;
	for (int index = 0; index < 3; index++)
	{
		int face = faces2check[index];

		int shift_face = (face-1)*(N+2)*(N+2);
		double value = -1;

		switch( face )
		{
			case 1:
				// face 3: Y = N+1
				MPI_Isend( array3d + (N+1)*(N+2), 1, datatype_faces[2], opposite_rank[2], 0, MPI_COMM_WORLD, &(my_requests[1]) );
				// face 1: Y = 0
				MPI_Irecv( recvarray, (N+2)*(N+2), MPI_DOUBLE, opposite_rank[0], 0, MPI_COMM_WORLD, &(my_requests[0]) );
				MPI_Waitall( 2, my_requests, MPI_STATUSES_IGNORE );

				if (opposite_rank[face-1] != MPI_PROC_NULL){
					#pragma omp parallel for private(value)
					for (int ind1 = 0; ind1 < N+2; ind1++){
						for (int ind2 = 0; ind2 < N+2; ind2++){
							value = recvarray[ind2 + ind1*(N+2) + shift_face];
							fill_array3d( array3d, N, ind2, 0, ind1, value );}
					}
				}
				break;
			case 4:
				// face 2: X = N+1
				MPI_Isend( array3d + N+1, 1, datatype_faces[1], opposite_rank[1], 0, MPI_COMM_WORLD, &(my_requests[0]) );
				// face 4: X = 0
				MPI_Irecv( recvarray + 3*(N+2)*(N+2), (N+2)*(N+2), MPI_DOUBLE, opposite_rank[3], 0, MPI_COMM_WORLD, &(my_requests[1]) );
				MPI_Waitall( 2, my_requests, MPI_STATUSES_IGNORE );

				if (opposite_rank[face-1] != MPI_PROC_NULL){
					#pragma omp parallel for private(value)
					for (int ind1 = 0; ind1 < N+2; ind1++){
						for (int ind2 = 0; ind2 < N+2; ind2++){
							value = recvarray[ind2 + ind1*(N+2) + shift_face];
							fill_array3d( array3d, N, 0, ind2, ind1, value );}
					}
				}
				break;
			case 6:
				// face 5: Z = N+1
				MPI_Isend( array3d + (N+1)*(N+2)*(N+2), 1, datatype_faces[4], opposite_rank[4], 0, MPI_COMM_WORLD, &(my_requests[0]) );
				// face 6: Z = 0
				MPI_Irecv( recvarray + 5*(N+2)*(N+2), (N+2)*(N+2), MPI_DOUBLE, opposite_rank[5], 0, MPI_COMM_WORLD, &(my_requests[1]) );
				MPI_Waitall( 2, my_requests, MPI_STATUSES_IGNORE );

				if (opposite_rank[face-1] != MPI_PROC_NULL){
					#pragma omp parallel for private(value)
					for (int ind1 = 0; ind1 < N+2; ind1++){
						for (int ind2 = 0; ind2 < N+2; ind2++){
							value = recvarray[ind2 + ind1*(N+2) + shift_face];
							fill_array3d( array3d, N, ind2, ind1, 0, value );}
					}
				}
				break;
			default:
				MPI_Abort( MPI_COMM_WORLD, -66 );
		}
	}

	return array3d;
}


double* initialize_array3d_cst( const int& N, const double& value )
{
	int sizearray3d = (N+2)*(N+2)*(N+2);
	double* array3d = new double[sizearray3d];
	#pragma omp parallel for
	for (int ii = 0; ii < sizearray3d; ii++){	array3d[ii] = value;}

	return array3d;
}


void print_slice( const double* const array3d, const int& N, 
		const int& direction, const int& index )
{
	switch (direction)
	{
		case 0:
			for (int jj = 0; jj < N+2; jj++){
				for (int kk = 0;  kk < N+2; kk++){
					printf("array3d[%d][%d][%d] = %f.\n", index, jj, kk, 
					evaluate_array3d( array3d, N, index, jj, kk ));}}
			break;

		case 1:
			for (int ii = 0; ii < N+2; ii++){
				for (int kk = 0;  kk < N+2; kk++){
					printf("array3d[%d][%d][%d] = %f.\n", ii, index, kk, 
					evaluate_array3d( array3d, N, ii, index, kk ));}}
			break;

		case 2:
			for (int ii = 0; ii < N+2; ii++){
				for (int jj = 0;  jj < N+2; jj++){
					printf("array3d[%d][%d][%d] = %f.\n", ii, jj, index,  
					evaluate_array3d( array3d, N, ii, jj, index ));}}
			break;
	}
}


void print_slicef( const double* const array3d, const int& N, 
		const int& direction, const int& index )
{
	switch (direction)
	{
		case 0:
			for (int kk = N+1;  kk >= 0; kk--){
				for (int jj = 0; jj < N+2; jj++){
					std::cout << evaluate_array3d( array3d, N, index, jj, kk ) << " ";}
				std::cout << std::endl;
			}
			break;

		case 1:
			for (int kk = N+1;  kk >= 0; kk--){
				for (int ii = 0; ii < N+2; ii++){
					std::cout << evaluate_array3d( array3d, N, ii, index, kk ) << " ";}
				std::cout << std::endl;
			}
			break;

		case 2:
			for (int jj = N+1;  jj >= 0; jj--){
				for (int ii = 0; ii < N+2; ii++){
					std::cout << evaluate_array3d( array3d, N, ii, jj, index ) << " ";}
				std::cout << std::endl;
			}
			break;
	}
}


void print_allslices( const double* const array3d, const int& N )
{
	for (int ii = 0; ii < N+2; ii++){
		printf( "\nPrint slice %d in direction 0.\n", ii );
		print_slice( array3d, N, 0, ii );}
}


void print_allslicesf( const double* const array3d, const int& N, const int& direction )
{
	for (int ii = 0; ii < N+2; ii++){
		printf( "\nPrint slice %d in direction %d.\n", ii, direction );
		print_slicef( array3d, N, direction, ii );}
}


/*This funciton is used to subtract an "array3d" vector from a "3d" function (output = fluxfct - array_3d_1) .. */
double* vecSub(const double* const array3d_1, double (*fluxfct) (const int&, const int&, const int&, const int&), const int& N)
{
	double* output = new double[(N+2)*(N+2)*(N+2)];
	double result;

	#pragma omp parallel for private(result)
	for(int kk = 0; kk < N+2; kk++)
	{
		for(int jj = 0; jj < N+2; jj++)
		{
			for(int ii = 0; ii < N+2; ii++)
			{
				result = fluxfct( N, ii, jj, kk) - evaluate_array3d( array3d_1, N, ii, jj, kk) ;
				fill_array3d( output, N, ii, jj, kk, result);
			}
		}
	}
	return output;
}


/*This funciton is used to subtract an "array3d" vector from a "3d" function (output = fluxfct - array_3d_1) .. */
double* vecSub_fun3d(const double* const array3d_1, const double* const fluxfct, const int& N)
{
	double* output = new double[(N+2)*(N+2)*(N+2)];
	double result;
	int ii, jj;

	#pragma omp parallel for private(result, ii, jj)
	for(int kk = 0; kk < N+2; kk++){
		for(jj = 0; jj < N+2; jj++){
			for(ii = 0; ii < N+2; ii++){
				result = evaluate_array3d( fluxfct, N, ii, jj, kk ) - evaluate_array3d( array3d_1, N, ii, jj, kk );
				fill_array3d( output, N, ii, jj, kk, result );}
		}
	}
	return output;
}


/*This funciton is used to subtract an "array3d" vector from a "3d" function (output = fluxfct - array_3d_1) .. */
void vecSub_fun3d_buff(double* const output, const double* const array3d_1, double* fluxfct, const int& N)
{
	double result;
	int ii, jj;

	#pragma omp parallel for private(result, ii, jj)
	for(int kk = 0; kk < N+2; kk++){
		for(jj = 0; jj < N+2; jj++){
			for(ii = 0; ii < N+2; ii++){
				result = evaluate_array3d( fluxfct, N, ii, jj, kk ) - evaluate_array3d( array3d_1, N, ii, jj, kk );
				fill_array3d( output, N, ii, jj, kk, result );}
		}
	}
}


/*This function returns the addtion of two "array3d" vectors*/
double* vecAdd(const double* const array3d_1, const double* const array3d_2, const int& N)
{
	double* output = new double[(N+2)*(N+2)*(N+2)];
	double result;

	#pragma omp parallel for private(result)
	for(int kk = 0; kk < N+2; kk++)
	{
		for(int jj = 0; jj < N+2; jj++)
		{
			for(int ii = 0; ii < N+2; ii++)
			{
				result = evaluate_array3d( array3d_1, N, ii, jj, kk) + evaluate_array3d( array3d_2, N, ii, jj, kk) ;
				fill_array3d( output, N, ii, jj, kk, result);
			}
		}
	}
	return output;
}


/* This function returns the addtion of two "array3d" vectors
Buffered version of the previous one. It updates the first argument directly. */
void vecAdd_buff(double*& array3d_1, const double* const array3d_2, double*& buffer, const int& N)
{
	double result;

	#pragma omp parallel for private(result)
	for(int kk = 0; kk < N+2; kk++)
	{
		for(int jj = 0; jj < N+2; jj++)
		{
			for(int ii = 0; ii < N+2; ii++)
			{
				result = evaluate_array3d( array3d_1, N, ii, jj, kk) + evaluate_array3d( array3d_2, N, ii, jj, kk);
				fill_array3d( buffer, N, ii, jj, kk, result );
			}
		}
	}

	double* tmp = array3d_1;
	array3d_1 = buffer;
	buffer = tmp;
}


// Copy a slice of an array3d
void copy_slice( double* const slice, const double* const array3d, const int& N, const int& slice_direction, const int& slice_index )
{
	switch (slice_direction)
	{
		case 1:	// Slice in plane x-y; index = z-coordinate
			#pragma omp parallel for
			for (int jj = 0; jj < N+2; jj++){
				for (int ii = 0; ii < N+2; ii++){
					slice[ii + jj*(N+2)] = array3d[ii + jj*(N+2) + slice_index*(N+2)*(N+2)];}
			}
			break;

		case 2:	// Slice in plane y-z; index = x-coordinate
			#pragma omp parallel for
			for (int jj = 0; jj < N+2; jj++){
				for (int ii = 0; ii < N+2; ii++){
					slice[ii + jj*(N+2)] = array3d[slice_index + ii*(N+2) + jj*(N+2)*(N+2)];}
			}
			break;

		case 3:	// Slice in plane x-z; index = y-coordinate
			#pragma omp parallel for
			for (int jj = 0; jj < N+2; jj++){
				for (int ii = 0; ii < N+2; ii++){
					slice[ii + jj*(N+2)] = array3d[ii + slice_index*(N+2) + jj*(N+2)*(N+2)];}
			}
			break;

		default:
			std::cout << "Error in copy_slice(...): slice_direction must be either 1, 2 or 3."
				  << std::endl;
			MPI_Abort( MPI_COMM_WORLD, -89 );
	}
}
