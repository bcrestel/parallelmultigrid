#include <stdio.h>
#include <cstdlib>
#include <omp.h>

#include "misc.h"
#include "boundarycondition3d.h"
#include "array3d.h"
#include "communication.h"


void restrict_3d_7pnt( double*& array3d, int& N, double* recvarray, const int* const opposite_rank, MPI_Datatype* datatype_faces )
{
	int new_N = (N-1)/2;
	double* new_X = new double[(new_N+2)*(new_N+2)*(new_N+2)];

	fill_recvarray( array3d, N, recvarray, opposite_rank, datatype_faces );
	
	double temp = 0;

	/* Determine bounds of indices that should be updated.
	Points on the boundary are not updated. */
	int kk_min = ( opposite_rank[5] == MPI_PROC_NULL ) ? 2 : 0; 	// Face 6
	int kk_max = ( opposite_rank[4] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 5
	int jj_min = ( opposite_rank[0] == MPI_PROC_NULL ) ? 2 : 0;	// Face 1
	int jj_max = ( opposite_rank[2] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 3
	int ii_min = ( opposite_rank[3] == MPI_PROC_NULL ) ? 2 : 0;	// Face 4
	int ii_max = ( opposite_rank[1] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 2


	#pragma omp parallel for private(temp)
	for(int k = kk_min; k < kk_max; k+=2){
		for(int j = jj_min; j < jj_max; j+=2){
			for(int i = ii_min; i < ii_max; i+=2){			
				temp = 0.5 * evaluate_array3d_bndy(array3d, N, i, j, k, recvarray);
				temp += (1.0/12.0) * (evaluate_array3d_bndy(array3d, N, i+1, j, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i-1, j, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j+1, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j-1, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j, k+1, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j, k-1, recvarray));

				fill_array3d(new_X, new_N, (i/2), (j/2), (k/2), temp);
			}
		}
	}

	// Fill the boundary points where needed
	if (opposite_rank[0] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk+=2){
			for (int ii = 0; ii < N+2; ii+=2){
				fill_array3d( new_X, new_N, (ii/2), 0, (kk/2), boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[1] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk+=2){
			for (int jj = 0; jj < N+2; jj+=2){
				fill_array3d( new_X, new_N, (N/2)+1, (jj/2), (kk/2), boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[2] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk+=2){
			for (int ii = 0; ii < N+2; ii+=2){
				fill_array3d( new_X, new_N, (ii/2), (N/2)+1, (kk/2), boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[3] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk+=2){
			for (int jj = 0; jj < N+2; jj+=2){
				fill_array3d( new_X, new_N, 0, (jj/2), (kk/2), boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[4] == MPI_PROC_NULL){
		for (int jj = 0; jj < N+2; jj+=2){
			for (int ii = 0; ii < N+2; ii+=2){
				fill_array3d( new_X, new_N, (ii/2), (jj/2), (N/2)+1, boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[5] == MPI_PROC_NULL){
		for (int jj = 0; jj < N+2; jj+=2){
			for (int ii = 0; ii < N+2; ii+=2){
				fill_array3d( new_X, new_N, (ii/2), (jj/2), 0, boundarycondition3d((N/2),0,0,0) );}
		}
	}

	
	// make array3d point to output data:
	delete[] array3d;
	array3d = new_X;

	N = new_N;
}


void restrict_3d_7pnt_buff( double*& array3d, double*& new_X, int& N, double* recvarray, const int* const opposite_rank, MPI_Datatype* datatype_faces )
{
	fill_recvarray( array3d, N, recvarray, opposite_rank, datatype_faces );
	
	double temp = 0;

	/* Determine bounds of indices that should be updated.
	Points on the boundary are not updated. */
	int kk_min = ( opposite_rank[5] == MPI_PROC_NULL ) ? 2 : 0; 	// Face 6
	int kk_max = ( opposite_rank[4] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 5
	int jj_min = ( opposite_rank[0] == MPI_PROC_NULL ) ? 2 : 0;	// Face 1
	int jj_max = ( opposite_rank[2] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 3
	int ii_min = ( opposite_rank[3] == MPI_PROC_NULL ) ? 2 : 0;	// Face 4
	int ii_max = ( opposite_rank[1] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 2


	#pragma omp parallel for private(temp)
	for(int k = kk_min; k < kk_max; k+=2)
	{
		for(int j = jj_min; j < jj_max; j+=2)
		{
			for(int i = ii_min; i < ii_max; i+=2)
			{			
				temp = 0.5 * evaluate_array3d_bndy(array3d, N, i, j, k, recvarray);
				temp += (1.0/12.0) * (evaluate_array3d_bndy(array3d, N, i+1, j, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i-1, j, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j+1, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j-1, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j, k+1, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j, k-1, recvarray));

				fill_array3d(new_X, (N/2), (i/2), (j/2), (k/2), temp);
			}
		}
	}

	// Fill the boundary points where needed
	if (opposite_rank[0] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk+=2){
			for (int ii = 0; ii < N+2; ii+=2){
				fill_array3d( new_X, (N/2), (ii/2), 0, (kk/2), boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[1] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk+=2){
			for (int jj = 0; jj < N+2; jj+=2){
				fill_array3d( new_X, (N/2), (N/2)+1, (jj/2), (kk/2), boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[2] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk+=2){
			for (int ii = 0; ii < N+2; ii+=2){
				fill_array3d( new_X, (N/2), (ii/2), (N/2)+1, (kk/2), boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[3] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk+=2){
			for (int jj = 0; jj < N+2; jj+=2){
				fill_array3d( new_X, (N/2), 0, (jj/2), (kk/2), boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[4] == MPI_PROC_NULL){
		for (int jj = 0; jj < N+2; jj+=2){
			for (int ii = 0; ii < N+2; ii+=2){
				fill_array3d( new_X, (N/2), (ii/2), (jj/2), (N/2)+1, boundarycondition3d((N/2),0,0,0) );}
		}
	}
	if (opposite_rank[5] == MPI_PROC_NULL){
		for (int jj = 0; jj < N+2; jj+=2){
			for (int ii = 0; ii < N+2; ii+=2){
				fill_array3d( new_X, (N/2), (ii/2), (jj/2), 0, boundarycondition3d((N/2),0,0,0) );}
		}
	}

	
	// Exchange pointer locations between array3d and buffer
	double* tmp = array3d;
	array3d = new_X;
	new_X = tmp;

	N = (N-1)/2;
}


// This method is not correct. we need to get values from processes on the corner.
void restrict_3d_19pnt( double*& array3d, int& N, double* recvarray, const int* const opposite_rank, MPI_Datatype* datatype_faces )
{
	fill_recvarray( array3d, N, recvarray, opposite_rank, datatype_faces );
	
	int dim = ((N+1)/2)+1;    // remember that the original dim = (N+2)
	int new_N  =  dim*dim*dim;
	double *new_X = new double[new_N];
	double temp = 0;

	/* Note that we need to restrict ghost values & global boundaries as well. So, there is no need to to determine bounds of indices that should be updated. */
	#pragma omp parallel for private(temp)
	for(int k = 0; k <= N+1; k+=2)
	{
		for(int j = 0; j <= N+1; j+=2)
		{
			for(int i = 0; i <= N+1; i+=2)
			{			
				temp = 0.5 * evaluate_array3d_bndy(array3d, N, i, j, k, recvarray);
				temp += (1.0/24.0) * (evaluate_array3d_bndy(array3d, N, i+1, j, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i-1, j, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j+1, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j-1, k, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j, k+1, recvarray) 
						    + evaluate_array3d_bndy(array3d, N, i, j, k-1, recvarray));

				temp += (1.0/48.0) * ( evaluate_array3d_bndy(array3d, N, i+1, j+1, k, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i-1, j+1, k, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i+1, j-1, k, recvarray) 
					 	     + evaluate_array3d_bndy(array3d, N, i-1, j-1, k, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i+1, j, k+1, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i-1, j, k+1, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i, j+1, k+1, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i, j-1, k+1, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i+1, j, k-1, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i-1, j, k-1, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i, j+1, k-1, recvarray) 
						     + evaluate_array3d_bndy(array3d, N, i, j-1, k-1, recvarray));

				fill_array3d(new_X, (N/2), (i/2), (j/2), (k/2), temp);
			}
		}
	}

	double *old_ref = array3d;
	array3d = new_X;
	delete[] old_ref;
	N = (N-1)/2;
}



