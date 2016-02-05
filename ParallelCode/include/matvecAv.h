#ifndef MATVECAV_INCLUDED
#define MATVECAV_INCLUDED

#include <mpi.h>

void matvecAv( double* vector_out, double* array3d, const int& N, double* recvarray, const int* const opposite_rank, MPI_Datatype* datatype_faces );

#endif
