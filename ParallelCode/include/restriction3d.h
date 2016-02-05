#ifndef RESTRICTION_INCLUDED
#define RESTRICTION_INCLUDED


void restrict_3d_7pnt( double*& array3d, int& N, double* recvarray, const int* const opposite_rank, MPI_Datatype* datatype_faces );

void restrict_3d_7pnti_buff( double*& array3d, double*& new_X, int& N, double* recvarray, const int* const opposite_rank, MPI_Datatype* datatype_faces );

//This is not correct ! (needs values from corner processes)
void restrict_3d_19pnt( double*& array3d, int& N, double* recvarray, const int* const opposite_rank, MPI_Datatype* datatype_faces );

#endif
