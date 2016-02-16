#ifndef ARRAY3D_INCLUDED
#define ARRAY3D_INCLUDED


void fill_array3d( double* const array, const int N, const int i, const int j, const int k, const double value );

double evaluate_array3d( const double* const array, const int N, const int i, const int j, const int k );

void print_slice( const double* const array3d, const int N, const int direction, const int index );

void print_sliceif( const double* const array3d, const int N, const int direction, const int index );

void print_allslices( const double* const array3d, const int N );

void print_allslicesf( const double* const array3d, const int N );

double* vecSub(const double* const array3d_1, double (*fluxfct) (const int, const int, const int, const int), const int N);

double* vecSub_fun3d(const double* const array3d_1, double* fluxfct, const int N);

double* vecAdd(const double* const array3d_1, const double* const array3d_2, const int N);

#endif

