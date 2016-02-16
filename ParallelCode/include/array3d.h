#ifndef ARRAY3D_INCLUDED
#define ARRAY3D_INCLUDED

#include <mpi.h>

#include "misc.h"
#include "communication.h"

double* initialize_array3d( const int N, double* recvarray, const int mpirank,
				const int* const opposite_rank, MPI_Datatype* datatype_faces );

double* initialize_array3d_cst( const int N, const double value );

void print_slice( const double* const array3d, const int N, const int direction, const int index );

/* Print in a formatted way (displayed like an actual face).
direction can be 0, 1 or 2 (x, y or z).
0 <= index <= N+1. */
void print_slicef( const double* const array3d, const int N, const int direction, const int index );

void print_allslices( const double* const array3d, const int N );

void print_allslicesf( const double* const array3d, const int N, const int direction = 0 );

double* vecSub(const double* const array3d_1, double (*fluxfct) (const int, const int, const int, const int), const int N);

double* vecSub_fun3d(const double* const array3d_1, const double* const fluxfct, const int N);

void vecSub_fun3d_buff(double* const output, const double* const array3d_1, double* fluxfct, const int N);

double* vecAdd(const double* const array3d_1, const double* const array3d_2, const int N);

void vecAdd_buff(double*& array3d_1, const double* const array3d_2, double*& buffer, const int N);

void copy_slice( double* const slice, const double* const array_3d, const int N, const int slice_direction, const int slice_index );

/* array3d contains the interior grid points (indices 1 -> N) and
the ghost values (indices 0 and N+1), for a total of (N+2)^3 pts. */
inline void fill_array3d( double* const array, const int N,
                const int i, const int j, const int k, const double value )
{
        if ( Is_inbetween( i, 0, N+1 ) && Is_inbetween( j, 0, N+1 ) && Is_inbetween( k, 0, N+1 ) ){
                array[ i + j*(N+2) + k*(N+2)*(N+2) ]  = value;}
        else{
                MPI_Abort( MPI_COMM_WORLD, -99 );}
}


inline double evaluate_array3d( const double* const array, const int N,
                        const int i, const int j, const int k )
{
        if ( Is_inbetween( i, 0, N+1 ) && Is_inbetween( j, 0, N+1 ) && Is_inbetween( k, 0, N+1 ) ){
                return ( array[ i + j*(N+2) + k*(N+2)*(N+2) ] );}
        else{
                MPI_Abort( MPI_COMM_WORLD, -88 );
                return 0.;
        }
}


/* Evaluate value of array and surrounding points as provided by
recvarray. Those recvarray points should have been recently updated
by fill_recvarray. */
inline double evaluate_array3d_bndy( const double* const array, const int N,
                                const int i, const int j, const int k,
                                const double* const recvarray )
{
        if ( Is_inbetween( i, 0, N+1 ) && Is_inbetween( j, 0, N+1 ) && Is_inbetween( k, 0, N+1 ) ){
                return array[ i + j*(N+2) + k*(N+2)*(N+2) ];}
        else{
                int my_face = what_face( i, j, k, N );
                int shift_face = (my_face-1)*(N+2)*(N+2);
                switch( my_face )
                {
                        case 1:
                                return recvarray[ i  + k*(N+2) + shift_face ];
                        case 2:
                                return recvarray[ j + k*(N+2) + shift_face ];
                        case 3:
                                return recvarray[ i  + k*(N+2) + shift_face ];
                        case 4:
                                return recvarray[ j  + k*(N+2) + shift_face ];
                        case 5:
                                return recvarray[ i  + j*(N+2) + shift_face ];
                        case 6:
                                return recvarray[ i  + j*(N+2) + shift_face ];
                        default:
                                MPI_Abort( MPI_COMM_WORLD, -77 );
                                return 0.;
                }
        }
}

#endif

