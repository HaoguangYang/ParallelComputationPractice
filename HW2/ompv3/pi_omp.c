#include <stdio.h>
#include <sys/time.h>

long long num_steps = 100000000;
double step;

int main(int argc, char* argv[])
{
	struct timeval start, stop;
	double x, pi, sum=0.0;
	int i,ret;
	step = 1./(double)num_steps;
	ret = gettimeofday(&start, NULL);
	if( ret != 0 ) printf("start timer error");
	int coreNum = omp_get_num_procs();
	double* sumArray = new double[coreNum];
	for (i=0; i<coreNum; i++)
		sumArray[i] = 0;

#pragma omp parallel for private(x)
	for (i=0; i<num_steps; i++)
	{
		int k = omp_get_thread_num();
		x = (i + .5)*step;
		sumArray[k] += 4.0/(1.+ x*x);
	}
	for (i=0; i<coreNum; i++)
		sum += sumArray[i];
	
	pi = sum*step;

	ret = gettimeofday(&stop, NULL);
	if( ret != 0 ) printf("stop timer error");

	printf("The value of PI is %20.17f\n",pi);
	printf("The time to calculate PI was %f microseconds\n",((double)(stop.tv_sec - start.tv_sec)*1000000.+(double)(stop.tv_usec - start.tv_usec)));
	return 0;
}
