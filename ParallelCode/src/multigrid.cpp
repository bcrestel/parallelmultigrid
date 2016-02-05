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

#include <mpi.h>
#include "communication.h"


/*This function implemented one v-cycle - serial function*/
//void v_cycle(double*& array3d, double h, int N, double* function_3d, int v1, int v2);



/* parallel setup & functions: */
void v_cycle_par(const int THRESHOLD, double*& array3d, int& N, const double* const fluxfct, 
			const int& v1, const int& v2, 
			double* recvarray, const int* const opposite_rank, 
			const double& omega, double* vector_out, bool gather = true );


int main(int argc, char **argv)
{
	// this code will be executed by all processes:

	// setting up MPI:
	MPI_Init(&argc, &argv);

	int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
	int* opposite_rank = opposite_procs( mpirank, mpisize );

	// Input parameters
	// Problem size:
	int N = (argc > 1) ? atoi(argv[1]) : 15; // MUST BE OF THE FORM 2^k - 1.
	// Total number of grid points for which we gather all data on rank 0
	const int THRESHOLD = (argc > 2) ? atoi(argv[2]) : (N-1)/2;	
	// Number of iterations:
	const int iter = (argc > 3) ? atoi(argv[3]) : 10;

	// Create input arrays needed for MG:
	double* recvarray = new double[6*(N+2)*(N+2)];	// buffer to receive info from neighbors.
	double* fluxfct = initialize_array3d_cst( N, 0.0 );	// right-hand side: -D u = f.
	double* vector_out = new double[(N+2)*(N+2)*(N+2)];	// buffer for result of matvec.
	MPI_Datatype* datatype_faces = get_datatypes( N );

	double* array3d = initialize_array3d( N, recvarray, mpirank, opposite_rank, datatype_faces );
	for (int ty = 0; ty < 6; ty++){	MPI_Type_free( &(datatype_faces[ty]) );}
	delete[] datatype_faces;

	int num_threads;
	#pragma omp parallel
	{
		#pragma omp master
		num_threads = omp_get_num_threads();
	}

	double omega = 2./3.;
	int v1 = 5;
	int v2 = 5;
	double t1, t2, tt = 0.0;

	// Start computation of Multigrid:
	if (mpirank == 0){	printf("MG with total pb size = %d, split over %d processes.\n\n", N*N*N*mpisize, mpisize);}
	double norm_2 = norm2_omp(array3d, (N+2)*(N+2)*(N+2));
	printf("v-cycle #0, 2-norm = %1f\n", norm_2);

	for (int ii = 0; ii < iter; ii++)
	{
		t1 = gtod_timer();
		v_cycle_par(THRESHOLD, array3d, N, fluxfct, v1, v2, recvarray, opposite_rank, omega, vector_out );
		t2 = gtod_timer();
		tt += t2 - t1;

		norm_2 = norm2_omp(array3d, (N+2)*(N+2)*(N+2));
		printf("v-cycle #%d, time = %f, 2-norm = %1f\n", ii+1, t2-t1, norm_2);
	}
	
	printf("Process %d, local pb size N = %d, Nb threads = %d, total nb iteration = %d, 2-norm = %e, inf-norm = %e, total run time = %f (msec).\n",
		mpirank, N, num_threads, iter, norm2_omp(array3d, (N+2)*(N+2)*(N+2)), norminf_omp(array3d, (N+2)*(N+2)*(N+2)), tt);

	delete[] array3d;
	delete[] fluxfct;
	delete[] vector_out;
	delete[] recvarray;
	delete[] opposite_rank;

	MPI_Finalize();
	
        return 0;
}



/* We use recursive calls to multigrid to go down the grid hierachy. Basic structure is as follows:
1) Smooth
2) Compute residual
3) Restrict residual to coarser mesh
4) Solve error equation A.e = r, using multigrid (recursive call)
5) Prolongate error
6) Correct solution: v <- v + e
7) Smooth 

Since we're doing that in parallel, at some point the number of grid points on a single process
is too small to justify the use of parallel communications. We gather all grid points on rank 0
and continue the coarsening, then prolongate until we reach the same number of elements
where we scatter the grid points.
*/

void v_cycle_par( const int THRESHOLD, double*& array3d,  int& N, const double* const fluxfct, const int& v1, const int& v2, 
		double* recvarray, const int* const opposite_rank, const double& omega, 
		double* vector_out, bool gather )
{
	// Define datatypes for that grid hierarchy:
	MPI_Datatype* datatype_faces = get_datatypes( N );

	// base case
	if(N == 1)
	{
		// Compute 'exact' solution with Jacobi (omega = 1)
		smoother(array3d, N, fluxfct, recvarray, opposite_rank, datatype_faces, 1.);
	}
	else
	{
		//*************************************************************************************************
		// (1) pre-smooth (v1) times:
		for(int i = 0; i < v1; i++){
			smoother(array3d, N, fluxfct, recvarray, opposite_rank, datatype_faces, omega);}

		//*************************************************************************************************
		// (2) compute residual vector:
		matvecAv( vector_out, array3d, N, recvarray, opposite_rank, datatype_faces );
		double* residual = vecSub_fun3d( vector_out, fluxfct, N );

		//*************************************************************************************************
		// (3) restrict residual:
		restrict_3d_7pnt( residual, N, recvarray, opposite_rank, datatype_faces );   // N will become: (N-1)/2

		//*************************************************************************************************
		// (4) recurse:
		double* e_2h;

		if ((N == THRESHOLD) && (gather))
		{
			int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
			int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

			int size_per_dim = find_cubicroot( mpisize );
			int N0 = (N+1)*size_per_dim - 1;

			int oldsize = (N+2)*(N+2)*(N+2);
			int newsize = (N0+2)*(N0+2)*(N0+2);

			// Define datatype for gather operation
			MPI_Datatype datatype_gather;
			MPI_Type_contiguous( oldsize, MPI_DOUBLE, &datatype_gather );
			MPI_Type_commit( &datatype_gather );

			double* recv_residual = new double[mpisize*oldsize];

			// Gather all data onto process 0: residual
			MPI_Gather( residual, 1, datatype_gather, recv_residual, 1, datatype_gather, 0, MPI_COMM_WORLD );
			delete[] residual;

			// Continue computation only on process 0
			if (mpirank == 0){
				// Re-define the neighbors of process 0 (no-one)
				int* opposite_rank_single = opposite_procs( 0, 1 );

				// Arrange recv_residual in a residual format:
				double* residual_total = new double[newsize];
				#pragma omp parallel for
				for (int rank = 0; rank < mpisize; rank++){
					int shift_ii = (rank % size_per_dim)*(N+1);
					int shift_jj = ((rank/size_per_dim)%size_per_dim)*(N+1);
					int shift_kk = rank/(size_per_dim*size_per_dim)*(N+1);
					for (int kk = 0; kk < N+2; kk++){
						for (int jj = 0; jj < N+2; jj++){
							for (int ii = 0; ii < N+2; ii++){
								residual_total[ii+shift_ii + (jj+shift_jj)*(N0+2) + (kk+shift_kk)*(N0+2)*(N0+2)] = 
											recv_residual[(ii + jj*(N+2) + kk*(N+2)*(N+2)) + rank*oldsize];
							}
						}
					}
				}

				double* vector_out_big = new double[newsize];
				e_2h = initialize_array3d_cst( N0, 0.0 ); // Initialize the error with 0.

				// Recurse on process 0
				v_cycle_par( THRESHOLD, e_2h, N0, residual_total, v1, v2, recvarray, opposite_rank_single, omega, vector_out_big, false );

				delete[] residual_total;
				delete[] vector_out_big;

				// Arrange e_2h in a convenient format
				#pragma omp parallel for
				for (int rank = 0; rank < mpisize; rank++){
					int shift_ii = (rank % size_per_dim)*(N+1);
					int shift_jj = ((rank/size_per_dim)%size_per_dim)*(N+1);
					int shift_kk = rank/(size_per_dim*size_per_dim)*(N+1);
					for (int kk = 0; kk < N+2; kk++){
						for (int jj = 0; jj < N+2; jj++){
							for (int ii = 0; ii < N+2; ii++){
								recv_residual[(ii + jj*(N+2) + kk*(N+2)*(N+2)) + rank*oldsize] = 
								e_2h[ii+shift_ii + (jj+shift_jj)*(N0+2) + (kk+shift_kk)*(N0+2)*(N0+2)];
							}
						}
					}
				}

				delete[] e_2h;
			}

			e_2h = new double[oldsize];
			
			// Scatter e_2h
			MPI_Scatter( recv_residual, 1, datatype_gather, e_2h, 1, datatype_gather, 0, MPI_COMM_WORLD );
			delete[] recv_residual;
			MPI_Type_free( &datatype_gather );
		}
		else{
			e_2h = initialize_array3d_cst( N, 0.0 ); // Initialize the error with 0.
			v_cycle_par( THRESHOLD, e_2h, N, residual, v1, v2, recvarray, opposite_rank, omega, vector_out, gather );
			delete[] residual;
		}

		//*************************************************************************************************
		// (5) prolongate:
		prolongation3d_par( e_2h, N );  // N will become 2*N + 1

		//*************************************************************************************************
		// (6) Correct:
		double* corrected_array3d =  vecAdd( array3d, e_2h, N );
		delete[] e_2h;
		delete[] array3d;
		array3d = corrected_array3d;

		//*************************************************************************************************
		// (7) post-smooth (v2) times ..
		for(int i = 0; i < v2; i++){
			smoother(array3d, N, fluxfct, recvarray, opposite_rank, datatype_faces, omega);}
	}
	for (int type = 0; type < 6; type++){	MPI_Type_free( &(datatype_faces[type]) );}
	delete[] datatype_faces;
}
