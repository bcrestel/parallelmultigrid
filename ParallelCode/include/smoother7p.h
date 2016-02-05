#ifndef SMOOTHERSEVEN_INCLUDED
#define SMOOTHERSEVEN_INCLUDED

#include <mpi.h>

void smoother( double*& array3d, const int N, 
		const double* const fluxfct, double* recvarray, 
		const int* const opposite_rank, MPI_Datatype* datatype_faces, 
		double omega = 2./3. );


void smoother_buff( double*& array3d, double*& buffer, const int N, 
		double* fluxfct, double* recvarray, 
		const int* const opposite_rank, MPI_Datatype* datatype_faces, 
		double omega = 2./3. );

#endif
