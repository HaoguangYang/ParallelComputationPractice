#include <stdio.h>
#include <sys/time.h>

long long num_steps = 1000000000;
double step;

int main(int argc, char* argv[])
{
	struct timeval start, stop;
	double x, pi, sum=0.0;
	int i,ret;
	step = 1./(double)num_steps;
	ret = gettimeofday(&start, NULL);
	if( ret != 0 ) printf("start timer error");

#pragma omp parallel for private(x) reduction(+:sum)
	for (i=0; i<num_steps; i++)
	{
		x = (i + .5)*step;
		sum = sum + 4.0/(1.+ x*x);
	}
	
	pi = sum*step;

	ret = gettimeofday(&stop, NULL);
	if( ret != 0 ) printf("stop timer error");

	printf("The value of PI is %20.17f\n",pi);
	printf("The time to calculate PI was %f microseconds\n",((double)(stop.tv_sec - start.tv_sec)*1000000.+(double)(stop.tv_usec - start.tv_usec)));
	return 0;
}
