#include <stdio.h>
#include <sys/time.h>

long long num_steps = 1000000000;
double step;

int calc(void)
{
	struct timeval start, stop;
	double x, pi, sum=0.0;
	int i,j,ret;
	step = 1./(double)num_steps;
	ret = gettimeofday(&start, NULL);
	if( ret != 0 ) printf("start timer error");
	int coreNum = omp_get_num_procs();
	double sumArray[coreNum];
	long long stride[coreNum+1];
	stride[0]=0;
	for (i=0; i<coreNum; i++)
	{
		sumArray[i] = 0;
		stride[i+1] = num_steps*(i+1)/coreNum;
	}

#pragma omp parallel for private(x) reduction(+:sum)
	for (j=0; j<coreNum; j++)
	{
	for (i=stride[j]; i<stride[j+1]; i++)
	{
		x = (i + .5)*step;
		sumArray[j] += 4.0/(1.+ x*x);
	}
	sum += sumArray[j];
	}
	
	pi = sum*step;

	ret = gettimeofday(&stop, NULL);
	if( ret != 0 ) printf("stop timer error");

	printf("Size of problem is %d\n",num_steps);
	printf("The value of PI is %20.17f\n",pi);
	printf("The time to calculate PI was %f microseconds\n\n",((double)(stop.tv_sec - start.tv_sec)*1000000.+(double)(stop.tv_usec - start.tv_usec)));
	return 0;
}

int main(int argc, char* argv[])
{
	if (argc>1)
	{
		int j;
		for (j=1; j<argc; j++)
		{
			num_steps = atoll(argv[j]);
			step = 0.0;
			int errnum = calc();
		}
	}
	return 0;
}