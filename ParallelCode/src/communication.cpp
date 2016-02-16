/*This file contains function dealing with communication in-between 
processes for array3d.

The 6 faces are numbered as:
face 1: Y = -1
face 2: X = N
face 3: Y = N
face 4: X = -1
face 5: Z = N
face 6: Z = -1

LIST OF MPI_ABORT CODES:
-77 = find_cubicroot() function
-66 = opposite_proc() function
-55 = what_face() function
-44 = neighboring_face() function
*/

#include "communication.h"
#include "array3d.h"
#include "boundarycondition3d.h"



/* Execute communications between processes to fill
recvarray with values of neighboring cells. 
We do not exchange outside faces, but the ones just
inside.
recvarray has size 6*(N+2)*(N+2). */
void fill_recvarray( double* array3d, const int N, 
			double* recvarray,
			const int* const opposite_rank,
			MPI_Datatype* datatype_faces )
{
	/* Start send/recv.
	Each face has its own datatype that takes care of the jumps in array3d. We just have to point
	at the right initial value.
	We use non-blocking send/recv to synchronise communications among processes.*/
	MPI_Request     my_requests[12];

// WARNING: ALL COMMUNICATIONS HAVE TAG 0. MAYBE NOT THE SAFEST CHOICE.
//	face 1: Y = 1
	MPI_Isend( array3d + N+2, 1, datatype_faces[0], opposite_rank[0], 0, MPI_COMM_WORLD, &(my_requests[6]) );
        MPI_Irecv( recvarray, (N+2)*(N+2), MPI_DOUBLE, opposite_rank[0], 0, MPI_COMM_WORLD, &(my_requests[0]) );
	
//	face 2: X = N
	MPI_Isend( array3d + N, 1, datatype_faces[1], opposite_rank[1], 0, MPI_COMM_WORLD, &(my_requests[7]) );
        MPI_Irecv( recvarray + (N+2)*(N+2), (N+2)*(N+2), MPI_DOUBLE, opposite_rank[1], 0, MPI_COMM_WORLD, &(my_requests[1]) );

//	face 3: Y = N
	MPI_Isend( array3d + N*(N+2), 1, datatype_faces[2], opposite_rank[2], 0, MPI_COMM_WORLD, &(my_requests[8]) );
        MPI_Irecv( recvarray + 2*(N+2)*(N+2), (N+2)*(N+2), MPI_DOUBLE, opposite_rank[2], 0, MPI_COMM_WORLD, &(my_requests[2]) );

//	face 4: X = 1
	MPI_Isend( array3d + 1, 1, datatype_faces[3], opposite_rank[3], 0, MPI_COMM_WORLD, &(my_requests[9]) );
        MPI_Irecv( recvarray + 3*(N+2)*(N+2), (N+2)*(N+2), MPI_DOUBLE, opposite_rank[3], 0, MPI_COMM_WORLD, &(my_requests[3]) );

//	face 5: Z = N
	MPI_Isend( array3d + N*(N+2)*(N+2), 1, datatype_faces[4], opposite_rank[4], 0, MPI_COMM_WORLD, &(my_requests[10]) );
        MPI_Irecv( recvarray + 4*(N+2)*(N+2), (N+2)*(N+2), MPI_DOUBLE, opposite_rank[4], 0, MPI_COMM_WORLD, &(my_requests[4]) );

//	face 6: Z = 1
	MPI_Isend( array3d + (N+2)*(N+2), 1, datatype_faces[5], opposite_rank[5], 0, MPI_COMM_WORLD, &(my_requests[11]) );
        MPI_Irecv( recvarray + 5*(N+2)*(N+2), (N+2)*(N+2), MPI_DOUBLE, opposite_rank[5], 0, MPI_COMM_WORLD, &(my_requests[5]) );

//	MPI_Status	my_statuses[6];

	// Wait for completion of all communications
	MPI_Waitall( 12, my_requests, MPI_STATUSES_IGNORE );
}


/*Return the number of the process that shares same ghost values data
on a corresponding face.*/
int opposite_proc( const int face, const int mpirank, const int mpisize )
{
	int proc_per_dim = find_cubicroot( mpisize );
        int res1 = (mpirank+1) % proc_per_dim;
        int res2 = (mpirank/proc_per_dim + 1) % proc_per_dim;
        int res3 = (mpirank/(proc_per_dim*proc_per_dim) + 1) % proc_per_dim;

	int opposite_proc = -99;

	switch(face)
	{
		case 1:
			opposite_proc = (res2 == 1) ? MPI_PROC_NULL : mpirank - proc_per_dim; 
			break;
		case 2:
			opposite_proc = (res1 == 0) ? MPI_PROC_NULL : mpirank + 1;
			break;
		case 3:
			opposite_proc = (res2 == 0) ? MPI_PROC_NULL : mpirank + proc_per_dim; 
			break;
		case 4:
			opposite_proc = (res1 == 1) ? MPI_PROC_NULL : mpirank - 1;
			break;
		case 5:
			opposite_proc = (res3 == 0) ? MPI_PROC_NULL : mpirank + proc_per_dim*proc_per_dim;
			break;
		case 6:
			opposite_proc = (res3 == 1) ? MPI_PROC_NULL : mpirank - proc_per_dim*proc_per_dim;
			break;
		default:
			MPI_Abort( MPI_COMM_WORLD, -66 );
			break;
	}

	return opposite_proc;
}


/*Return all the processes that share a common face with
the current process. The ordering is similar to the one
for the faces.*/
int* opposite_procs( const int mpirank, const int mpisize )
{
	double t1, tt1, tt2;
	t1 = MPI_Wtime();

        int* opposite_rank = new int[6];        // Store the rank of processes 
                                                // that are opposite to a given face.
                                                // Face indices given by def at top.
	tt1 = MPI_Wtime();
        int proc_per_dim = find_cubicroot( mpisize );
	tt2 = MPI_Wtime(); tt2 = tt2 - tt1;
        int res1 = (mpirank+1) % proc_per_dim;
        int res2 = (mpirank/proc_per_dim + 1) % proc_per_dim;
        int res3 = (mpirank/(proc_per_dim*proc_per_dim) + 1) % proc_per_dim;

/*	if (mpirank == 0){ printf("proc_per_dim=%d, res1=%d, res2=%d, res3=%d\n", 
		proc_per_dim, res1, res2, res3);}*/

        /* Find ranks of surrounding proc
        Value of -1 means boundary
        The 6 faces are numbered as:
        face 1: Y = -1 (index 0)
        face 2: X = N  (index 1)
        face 3: Y = N  (index 2)
        face 4: X = -1 (index 3)
        face 5: Z = N  (index 4)
        face 6: Z = -1 (index 5)*/
        opposite_rank[0] = (res2 == 1) ? MPI_PROC_NULL : mpirank - proc_per_dim;
        opposite_rank[1] = (res1 == 0) ? MPI_PROC_NULL : mpirank + 1;
        opposite_rank[2] = (res2 == 0) ? MPI_PROC_NULL : mpirank + proc_per_dim;
        opposite_rank[3] = (res1 == 1) ? MPI_PROC_NULL : mpirank - 1;
        opposite_rank[4] = (res3 == 0) ? MPI_PROC_NULL : mpirank + proc_per_dim*proc_per_dim;
        opposite_rank[5] = (res3 == 1) ? MPI_PROC_NULL : mpirank - proc_per_dim*proc_per_dim;
	tt1 = MPI_Wtime();
	tt1 = tt1 - t1;

//	printf("tt1=%e, tt2=%e\n", tt1, tt2);

	return opposite_rank;
}


/* Define MPI_Datatype's that will be used
to pass array3d data to/from neighboring
processes.
Each MPI_Datatype correspond to a face, with
the numbering being the same as for the faces. */
MPI_Datatype* get_datatypes( const int N )
{
	MPI_Datatype* output = new MPI_Datatype[6];

	MPI_Type_vector( (N+2), N+2, (N+2)*(N+2), MPI_DOUBLE, &(output[0]) );
	MPI_Type_vector( (N+2)*(N+2), 1, N+2, MPI_DOUBLE, &(output[1]) );
	MPI_Type_vector( (N+2), N+2, (N+2)*(N+2), MPI_DOUBLE, &(output[2]) );
	MPI_Type_vector( (N+2)*(N+2), 1, N+2, MPI_DOUBLE, &(output[3]) );
	MPI_Type_contiguous( (N+2)*(N+2), MPI_DOUBLE, &(output[4]));
	MPI_Type_contiguous( (N+2)*(N+2), MPI_DOUBLE, &(output[5]));

	for (int ii = 0; ii < 6; ii++){
		MPI_Type_commit( &(output[ii]) );}

	return output;
}


/* return an array with the index of faces
around a given face
WARNING: THIS MUST BE CONSISTENT WITH THE
ORDERING FOR GHOST RELATIVELY TO ARRAY3D.
That is, we define all top, left, bottom, right 
when looking at the face 1 of the cube.*/
int* neighboring_face( const int face_index )
{
	int* output = new int[4];
	
	/* neighboring faces are presented in the following order:
	left, bottom, right, top*/

	switch( face_index )
	{
		case 1:
			output[0] = 4;
			output[1] = 6;
			output[2] = 2;
			output[3] = 5;
			break;
		case 2:
			output[0] = 1;
			output[1] = 6;
			output[2] = 3;
			output[3] = 5;
			break;
		case 3:
			output[0] = 4;
			output[1] = 6;
			output[2] = 2;
			output[3] = 5;
			break;
		case 4:
			output[0] = 1;
			output[1] = 6;
			output[2] = 3;
			output[3] = 5;
			break;
		case 5:
			output[0] = 4;
			output[1] = 1;
			output[2] = 2;
			output[3] = 3;
			break;
		case 6:
			output[0] = 4;
			output[1] = 1;
			output[2] = 2;
			output[3] = 3;
			break;
		default:
			MPI_Abort( MPI_COMM_WORLD, -44 );
			break;
	}

	return output;
}
