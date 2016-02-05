#include <stdlib.h>
#include <sys/time.h>


double gtod_timer()
{
    struct timeval tv;

   gettimeofday(&tv, NULL);

   return (double)tv.tv_sec * 1000 + (double)tv.tv_usec / 1000;
}

