#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "array3d.h"
#include "prolongation3d.h"
#include "misc.h"
#include "smoother7p.h"
#include "communication.h"
#include "matvecAv.h"
#include "restriction3d.h"

#include <mpi.h>

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


double fluxfunction( const int N, const int ii, const int jj, const int kk ){	return 0.;}

void test_smoother7p()
{
	int N = 3;
	double* array3d = new double[N*N*N];

	for (int ii = 0; ii < N*N*N; ii++){	array3d[ii] = 1.;}
	printf("\n\nPrint before smoothing.\n");
	print_allslicesf( array3d, N );

	printf("\n\nSmoothing.\n");
//	smoother( array3d, N, fluxfunction );

	printf("\n\nPrint after smoothing.\n");
	print_allslicesf( array3d, N );
}


void test_neighbors(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
	int* my_neighbors = opposite_procs( mpirank, mpisize );
	if (mpirank == 2){
		printf("Printing neighbors of process %d: ", mpirank);
		for (int ii = 0; ii < 6; ii++){	printf("%d ", my_neighbors[ii]);}
		printf("\n");
	}

	MPI_Finalize();
}


void test_initialize( int argc, char* argv[] )
{
	MPI_Init(&argc, &argv);

	int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

	int* my_neighbors = opposite_procs( mpirank, mpisize );
	int N = 2; 
	double* recvarray = new double[6*(N+2)*(N+2)];
	MPI_Datatype* my_datatypes = get_datatypes( N );
	double* array3d = initialize_array3d( N, recvarray, mpirank, my_neighbors, my_datatypes );

/*	for (int ii = 0; ii < mpisize; ii++)
	{
		MPI_Barrier( MPI_COMM_WORLD );
		if (mpirank == ii){ 
			for (int jj = 0; jj < 6; jj++){	printf("%d ", my_neighbors[jj]);}
			printf("\nPrint process %d.\n", mpirank);
			print_allslicesf( array3d, N, 0 );
			printf("\n\n");
		}
	}*/
	int proc_per_dim = find_cubicroot( mpisize );
	int target = proc_per_dim*proc_per_dim + proc_per_dim + 1;
	if (mpirank == target){ 
		printf("Process %d.\n", mpirank);
		print_slicef( array3d, N, 1, N+1 );
		printf("\n\n");}
	MPI_Barrier( MPI_COMM_WORLD );
	if (mpirank == target+1){ 
		printf("Process %d.\n", mpirank);
		print_slicef( array3d, N, 1, N+1 );
		printf("\n\n");}
	MPI_Barrier( MPI_COMM_WORLD );
	if (mpirank == target+proc_per_dim){ 
		printf("Process %d.\n", mpirank);
		print_slicef( array3d, N, 1, 0 );
		printf("\n\n");}
	MPI_Barrier( MPI_COMM_WORLD );
	if (mpirank == target+proc_per_dim*proc_per_dim){ 
		printf("Process %d.\n", mpirank);
		print_slicef( array3d, N, 1, N+1 );
		printf("\n\n");}

	delete[] my_neighbors;
	delete[] recvarray;
	delete[] my_datatypes;
	delete[] array3d;

	MPI_Finalize();
}


void test_fillrecvarray( int argc, char* argv[] )
{
	MPI_Init(&argc, &argv);

	int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

	int* opposite_rank = opposite_procs( mpirank, mpisize );
	int N = 1; 
	double* recvarray = new double[6*(N+2)*(N+2)];
	MPI_Datatype* datatype_faces = get_datatypes( N );
	double* array3d = initialize_array3d_cst( N, 1. );

	fill_recvarray( array3d, N, recvarray, opposite_rank, datatype_faces );

	if (mpirank == 21){ 
		printf("Process %d.\n", mpirank);
		for (int ii = 0; ii < 6*(N+2)*(N+2); ii++){	printf("%f \n", recvarray[ii]);}
		printf("\n\n");
		print_allslicesf( array3d, N );
	}

	MPI_Finalize();
}


void test_matvec( int argc, char* argv[] )
{
	MPI_Init(&argc, &argv);

	int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

	int* opposite_rank = opposite_procs( mpirank, mpisize );
	int N = 1; 
	double* recvarray = new double[6*(N+2)*(N+2)];
	MPI_Datatype* datatype_faces = get_datatypes( N );
	double* array3d = initialize_array3d_cst( N, 1. );
	double* vector_out = new double[(N+2)*(N+2)*(N+2)];

	matvecAv( vector_out, array3d, N, recvarray, opposite_rank, datatype_faces );

	if (mpirank == 21){ 
		printf("Process %d.\n", mpirank);
		for (int ii = 0; ii < 6*(N+2)*(N+2); ii++){	printf("%f \n", recvarray[ii]);}
		printf("\n\n");
		print_allslicesf( array3d, N );
		printf("\n\n");
		print_allslicesf( vector_out, N );}

	MPI_Finalize();
}


void test_restriction3d( int argc, char* argv[] )
{
	MPI_Init(&argc, &argv);

	int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

	int* opposite_rank = opposite_procs( mpirank, mpisize );
	int N = 3; 
	double* recvarray = new double[6*(N+2)*(N+2)];
	MPI_Datatype* datatype_faces = get_datatypes( N );
	double* array3d = initialize_array3d_cst( N, 1. );
	double* buffer = new double[(N+2)*(N+2)*(N+2)];

	//double* vector_out = new double[(N+2)*(N+2)*(N+2)];
	//matvecAv( vector_out, array3d, N, recvarray, opposite_rank, datatype_faces );
	
	int rnk = 0;	

	if(mpirank == rnk)
	{
		printf("\nProcess %d, N = %d.\n", mpirank, N);
		print_allslicesf( array3d, N );
		printf("\n************************************\n");
	}

	restrict_3d_7pnt( array3d, buffer, N, recvarray, opposite_rank, datatype_faces );

	if(mpirank == rnk)
	{
		printf("\nProcess %d, N = %d.\n", mpirank, N);
		print_allslicesf( array3d, N );
		printf("\n************************************\n");
	}

	MPI_Finalize();
}

void test_prolongation3d_par( int argc, char* argv[] )
{
	MPI_Init(&argc, &argv);

	int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

	int* opposite_rank = opposite_procs( mpirank, mpisize );
	int N = 7; 
	double* array3d = initialize_array3d_cst( N, 1. );
	double* buffer = new double[(N+2)*(N+2)*(N+2)];
	
	N=3;
	MPI_Datatype* datatype_faces = get_datatypes( N );

	//double* vector_out = new double[(N+2)*(N+2)*(N+2)];
	//matvecAv( vector_out, array3d, N, recvarray, opposite_rank, datatype_faces );
	
	int rnk = 7;	

	if(mpirank == rnk)
	{
		printf("\nProcess %d, N = %d.\n", mpirank, N);
		print_allslicesf( array3d, N );
		printf("\n************************************\n");
	}

	prolongation3d_par( array3d, buffer, N, opposite_rank, datatype_faces );

	if(mpirank == rnk)
	{
		printf("\nProcess %d, N = %d.\n", mpirank, N);
		print_allslicesf( array3d, N );
		printf("\n************************************\n");
	}

	delete[] opposite_rank;
	delete[] array3d;
	delete[] buffer;
	delete[] datatype_faces;

	MPI_Finalize();
}


void test_smoother( int argc, char* argv[] )
{
	MPI_Init(&argc, &argv);

	int mpirank; MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	int mpisize; MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

	int* opposite_rank = opposite_procs( mpirank, mpisize );
	int N = 5; 
	double* recvarray = new double[6*(N+2)*(N+2)];
	MPI_Datatype* datatype_faces = get_datatypes( N );
	double* array3d = initialize_array3d( N, recvarray, mpirank, opposite_rank, datatype_faces );
	//double* array3d = initialize_array3d_cst( N, 1. );
	double* buffer = new double[(N+2)*(N+2)*(N+2)];
	double* fluxfct = new double[(N+2)*(N+2)*(N+2)];
	for (int ii = 0; ii < (N+2)*(N+2)*(N+2); ii++){	fluxfct[ii] = 0.;}
	double omega = 2./3.;

	int printrank = 2;
	if (mpirank == printrank){ 
		printf("Process %d.\n", mpirank);
		print_allslicesf( array3d, N );}
	
	for (int ii = 0; ii < 1000; ii++){
	smoother( array3d, buffer, N, fluxfct, recvarray, opposite_rank, datatype_faces, omega );}

	if (mpirank == printrank){ 
		printf("Process %d.\n", mpirank);
		print_allslicesf( array3d, N );}

	MPI_Finalize();
}


void switch_array( double**& errors, double*& buffer, int index )
{
	double* tmp = buffer;
	buffer = errors[index];
	errors[index] = tmp;
}

void test_r2p2p()
{
	double** errors = new double*[3];
	for (int ii = 0; ii < 3; ii++){
		errors[ii] = new double[10];
		for (int jj = 0; jj < 10; jj++){
			errors[ii][jj] = (double)(ii);}
	}

	double* buffer = new double[10];
	for (int ii = 0; ii < 10; ii++){	buffer[ii] = 99.;}

	// Print buffer & errors
	for (int ii = 0; ii < 10; ii++){	printf("%f ", buffer[ii]);}
	printf("\n");
	for (int ii = 0; ii < 3; ii++){
		for (int jj = 0; jj < 10; jj++){	printf("%f ", errors[ii][jj]);}
		printf("\n");
	}
	printf("\n\n");

	// Switch columns
	switch_array(errors, buffer, 0);

	// Print buffer & errors
	for (int ii = 0; ii < 10; ii++){	printf("%f ", buffer[ii]);}
	printf("\n");
	for (int ii = 0; ii < 3; ii++){
		for (int jj = 0; jj < 10; jj++){	printf("%f ", errors[ii][jj]);}
		printf("\n");
	}
	printf("\n\n");
}


int main(int argc, char* argv[])
{
//	testarray3d();
//	test_prolongation();
//	test_misc();
//	test_smoother7p();
//	test_neighbors( argc, argv );
//	test_initialize( argc, argv );
//	test_matvec( argc, argv );
//	test_fillrecvarray( argc, argv );


//////////////////////////////	
//Testing Restriction & Prolongation:

//	test_restriction3d(argc, argv);
//	test_prolongation3d_par(argc, argv);

//	test_smoother(argc, argv);
	
	test_r2p2p();

	return 0;
}
