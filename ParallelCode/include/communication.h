#ifndef COMMUNICATION_INCLUDED
#define COMMUNICATION_INCLUDED

#include <mpi.h>

void fill_recvarray( double* array3d, const int N, double* recvarray, const int* const opposite_rank, MPI_Datatype* datatype_faces );

int opposite_proc( const int face, const int mpirank, const int mpisize );

int* opposite_procs( const int mpirank, const int mpisize );

MPI_Datatype* get_datatypes( const int N );

int* neighboring_face( const int face_index );

/*Simplified cubic root function to avoid
expensive function call.*/
inline int find_cubicroot( const int number )
{
        switch (number)
        {
                case 1:
                        return 1;
                        break;
                case 8:
                        return 2;
                        break;
                case 64:
                        return 4;
                        break;
                case 512:
                        return 8;
                        break;
                case 4096:
                        return 16;
                        break;
                case 32768:
                        return 32;
                        break;
                default:
                        MPI_Abort( MPI_COMM_WORLD, -77 );
                        return 0;
        }
}


/* Determine what face we are looking at based on the indices
when evaluating points outside of array3d. */
inline int what_face( const int i, const int j, const int k, const int N )
{
	if (i == -1){		return 4;}
	else if (i == N+2){	return 2;}
	else if (j == -1){	return 1;}
	else if (j == N+2){	return 3;}
	else if (k == -1){	return 6;}
	else if (k == N+2){	return 5;}
	else{	MPI_Abort( MPI_COMM_WORLD, -55 );
		return 0;}
}


#endif
