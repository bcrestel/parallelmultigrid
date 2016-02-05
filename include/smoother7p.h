#ifndef SMOOTHERSEVEN_INCLUDED
#define SMOOTHERSEVEN_INCLUDED

void smoother7p( double*& array3d, const int N, double (*fluxfct) (const int&, const int&, const int&, const int&), double omega = 2./3. );
void smoother7p_3dfun( double*& array3d, const int N, double* fluxfct, double omega = 2./3. );

#endif
