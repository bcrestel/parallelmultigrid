#ifndef MATHLOCAL_INCLUDED
#define MATHLOCAL_INCLUDED

#include <cmath>

double norm2_omp( const double* const V, const int Ntotal );

double norminf_omp( const double* const V, const int Ntotal );

inline bool isequal( const double d1, const double d2, const double precision = 1e-14 )
{	return ( fabs(d1 - d2) <= precision * fabs(d2) );}

inline bool Is_even( const unsigned int n )
{       return (n % 2 == 0);}

inline bool Is_inbetween( const int i, const int lb, const int up )
{       return ( (i >= lb) & (i <= up) );}

int index2unique( const int ii, const int jj, const int kk );

#endif
