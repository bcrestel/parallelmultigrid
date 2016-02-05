/* smoother7p: applies relaxation to a 1d-vector representing
a 3d-array, as defined in array3d.cpp, and using a
7-point stencil in finite-differences.
*/

#include <omp.h>

#include "array3d.h"
#include "communication.h"
#include "smoother7p.h"
#include "boundarycondition3d.h"

void smoother( double*& array3d, const int N, 
		const double* const fluxfct, double* recvarray,
		const int* const opposite_rank, MPI_Datatype* datatype_faces, 
		double omega )
{
	int sizebuff = (N+2)*(N+2)*(N+2);
	double* buffer = new double[sizebuff];

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


	double value, ngh;
	#pragma omp parallel 
	{
		#pragma omp for private( value, ngh ) nowait 
		for (int kk = kk_min; kk < kk_max; kk++){
			for (int jj = jj_min; jj < jj_max; jj++){
				for (int ii = ii_min; ii < ii_max; ii++){
					ngh = evaluate_array3d_bndy( array3d, N, ii-1, jj, kk, recvarray );
					ngh += evaluate_array3d_bndy( array3d, N, ii+1, jj, kk, recvarray );
					ngh += evaluate_array3d_bndy( array3d, N, ii, jj-1, kk, recvarray );
					ngh += evaluate_array3d_bndy( array3d, N, ii, jj+1, kk, recvarray );
					ngh += evaluate_array3d_bndy( array3d, N, ii, jj, kk-1, recvarray );
					ngh += evaluate_array3d_bndy( array3d, N, ii, jj, kk+1, recvarray );
					value = ( h*h* evaluate_array3d(fluxfct, N, ii, jj, kk ) + ngh ) / 6.;

					fill_array3d( buffer, N, ii, jj, kk, (1.-omega)*evaluate_array3d( array3d, N, ii, jj, kk ) + omega*value );}
			}
		}

		// Fill the boundary points where needed
		if (opposite_rank[0] == MPI_PROC_NULL){
			#pragma omp for nowait
			for (int kk = 0; kk < N+2; kk++){
				for (int ii = 0; ii < N+2; ii++){
					fill_array3d( buffer, N, ii, 0, kk, boundarycondition3d(N,0,0,0) );}
			}
		}
		if (opposite_rank[1] == MPI_PROC_NULL){
			#pragma omp for nowait
			for (int kk = 0; kk < N+2; kk++){
				for (int jj = 0; jj < N+2; jj++){
					fill_array3d( buffer, N, N+1, jj, kk, boundarycondition3d(N,0,0,0) );}
			}
		}
		if (opposite_rank[2] == MPI_PROC_NULL){
			#pragma omp for nowait
			for (int kk = 0; kk < N+2; kk++){
				for (int ii = 0; ii < N+2; ii++){
					fill_array3d( buffer, N, ii, N+1, kk, boundarycondition3d(N,0,0,0) );}
			}
		}
		if (opposite_rank[3] == MPI_PROC_NULL){
			#pragma omp for nowait
			for (int kk = 0; kk < N+2; kk++){
				for (int jj = 0; jj < N+2; jj++){
					fill_array3d( buffer, N, 0, jj, kk, boundarycondition3d(N,0,0,0) );}
			}
		}
		if (opposite_rank[4] == MPI_PROC_NULL){
			#pragma omp for nowait 
			for (int jj = 0; jj < N+2; jj++){
				for (int ii = 0; ii < N+2; ii++){
					fill_array3d( buffer, N, ii, jj, N+1, boundarycondition3d(N,0,0,0) );}
			}
		}
		if (opposite_rank[5] == MPI_PROC_NULL){
			#pragma omp for nowait 
			for (int jj = 0; jj < N+2; jj++){
				for (int ii = 0; ii < N+2; ii++){
					fill_array3d( buffer, N, ii, jj, 0, boundarycondition3d(N,0,0,0) );}
			}
		}
	}

	// Exchange pointer locations between array3d and buffer
	double* tmp = array3d;
	array3d = buffer;
	buffer = tmp;
}


void smoother_buff( double*& array3d, double*& buffer, const int N, 
		double* fluxfct, double* recvarray,
		const int* const opposite_rank, MPI_Datatype* datatype_faces, 
		double omega )
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

	double value, ngh;
	#pragma omp parallel for private( value, ngh )
	for (int kk = kk_min; kk < kk_max; kk++){
		for (int jj = jj_min; jj < jj_max; jj++){
			for (int ii = ii_min; ii < ii_max; ii++){
				ngh = evaluate_array3d_bndy( array3d, N, ii-1, jj, kk, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii+1, jj, kk, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii, jj-1, kk, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii, jj+1, kk, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii, jj, kk-1, recvarray );
				ngh += evaluate_array3d_bndy( array3d, N, ii, jj, kk+1, recvarray );
				value = ( h*h* evaluate_array3d(fluxfct, N, ii, jj, kk ) + ngh ) / 6.;

				fill_array3d( buffer, N, ii, jj, kk, (1.-omega)*evaluate_array3d( array3d, N, ii, jj, kk ) + omega*value );}
		}
	}

	// Fill the boundary points where needed
	if (opposite_rank[0] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk++){
			for (int ii = 0; ii < N+2; ii++){
				fill_array3d( buffer, N, ii, 0, kk, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[1] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk++){
			for (int jj = 0; jj < N+2; jj++){
				fill_array3d( buffer, N, N+1, jj, kk, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[2] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk++){
			for (int ii = 0; ii < N+2; ii++){
				fill_array3d( buffer, N, ii, N+1, kk, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[3] == MPI_PROC_NULL){
		for (int kk = 0; kk < N+2; kk++){
			for (int jj = 0; jj < N+2; jj++){
				fill_array3d( buffer, N, 0, jj, kk, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[4] == MPI_PROC_NULL){
		for (int jj = 0; jj < N+2; jj++){
			for (int ii = 0; ii < N+2; ii++){
				fill_array3d( buffer, N, ii, jj, N+1, boundarycondition3d(N,0,0,0) );}
		}
	}
	if (opposite_rank[5] == MPI_PROC_NULL){
		for (int jj = 0; jj < N+2; jj++){
			for (int ii = 0; ii < N+2; ii++){
				fill_array3d( buffer, N, ii, jj, 0, boundarycondition3d(N,0,0,0) );}
		}
	}

	// Exchange pointer locations between array3d and buffer
	delete[] array3d;
	array3d = buffer;
}
