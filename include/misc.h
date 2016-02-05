#ifndef MATHLOCAL_INCLUDED
#define MATHLOCAL_INCLUDED

double norm2_omp( const double* const V, const int& Ntotal );

double norminf_omp( const double* const V, const int& Ntotal );

bool isequal( const double& d1, const double& d2, const double precision );

bool Is_even( const unsigned int& n );

bool Is_inbetween( const int& i, const int& lb, const int& up );

int index2unique( const int& ii, const int& jj, const int& kk );

#endif
