/* matvecAv7p returns the matrix-vector product A.v, where
v is the solution vector for a 3d problem stored in a 1d-vector
as defined in array3d.cpp, and A is the 3D laplacian discretized
with finite-differences and a 7-point stencil.
N is the number of interior grid points along one dimension.
*/

#include <omp.h>

#include "array3d.h"
#include "communication.h"
#include "matvecAv.h"
#include "boundarycondition3d.h"


/* Compute a matvec with 7-pt stencil for A = 2nd-order centered finite-difference
scheme and vector = array3d. Results will contain as many points as array3d, i.e
(N+2)*(N+2)*(N+2). */
void matvecAv( double* vector_out, double* array3d, const int& N, double* recvarray,
					const int* const opposite_rank, MPI_Datatype* datatype_faces )
{
	fill_recvarray( array3d, N, recvarray, opposite_rank, datatype_faces );

	double h = 1. / (N + 1.);


	/* Determine bounds of indices that should be updated.
	Points on the boundary are not updated. */
	int kk_min = ( opposite_rank[5] == MPI_PROC_NULL ) ? 1 : 0; 	// Face 6
	int kk_max = ( opposite_rank[4] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 5
	int jj_min = ( opposite_rank[0] == MPI_PROC_NULL ) ? 1 : 0;	// Face 1
	int jj_max = ( opposite_rank[2] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 3
	int ii_min = ( opposite_rank[3] == MPI_PROC_NULL ) ? 1 : 0;	// Face 4
	int ii_max = ( opposite_rank[1] == MPI_PROC_NULL ) ? N+1 : N+2;	// Face 2

	double ngh, value;
	#pragma omp parallel for private(ngh, value)
	for (int kk = kk_min; kk < kk_max; kk++){
		for (int jj = jj_min; jj < jj_max; jj++){
			for (int ii = ii_min; ii < ii_max; ii++){
				ngh = evaluate_array3d_bndy( array3d, N, ii-1, jj, kk, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii+1, jj, kk, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii, jj-1, kk, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii, jj+1, kk, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii, jj, kk-1, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii, jj, kk+1, recvarray );
				value = ( 6. * evaluate_array3d( array3d, N, ii, jj, kk ) -  ngh ) / (h*h);
				
				fill_array3d( vector_out, N, ii, jj, kk, value );
			}
		}
	}

	// Fill the boundary points where needed
	if (opposite_rank[0] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk++){
			for (int ii = 0; ii < N+2; ii++){
				fill_array3d( vector_out, N, ii, 0, kk, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[1] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk++){
			for (int jj = 0; jj < N+2; jj++){
				fill_array3d( vector_out, N, N+1, jj, kk, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[2] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk++){
			for (int ii = 0; ii < N+2; ii++){
				fill_array3d( vector_out, N, ii, N+1, kk, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[3] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk++){
			for (int jj = 0; jj < N+2; jj++){
				fill_array3d( vector_out, N, 0, jj, kk, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[4] == MPI_PROC_NULL){
		for (int jj = 0; jj < N+2; jj++){
			for (int ii = 0; ii < N+2; ii++){
				fill_array3d( vector_out, N, ii, jj, N+1, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[5] == MPI_PROC_NULL){
		for (int jj = 0; jj < N+2; jj++){
			for (int ii = 0; ii < N+2; ii++){
				fill_array3d( vector_out, N, ii, jj, 0, boundarycondition3d(N,0,0,0) );}
		}
	}
}

