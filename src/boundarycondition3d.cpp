/* boundarycondition3d: Returns the value of the boundary condition at a certain grid point.
N = the number of grid points (stored) along one dimension, i.e the total size of the
3d array storing the interior grid points is N*N*N.
The entries ii, jj, kk should be either -1 or N (for at least one of the three indices).
Indices 0,...,N-1 are interior.
*/

double boundarycondition3d( const int N, const int ii, const int jj, const int kk )
{
	return 0.;
}
