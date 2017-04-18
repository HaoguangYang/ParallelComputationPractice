#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>

using namespace std;

#define NX 300
#define NY 300
#define NZ 300
#define BATCH  10

double a[NZ][NY][NX];
double b[NZ][NY][NX];

void test4(int S){
//Ordinary three-fold loop with ivdep and loop order optimized
	//double (*a)[NY][NX] = A;
    //double (*b)[NY][NX] = B;
	double division = 1/7;
    for(int step = 0; step < S; step ++){
        for (int k = 1; k < NX-1; k ++){
            for(int j = 1; j < NY-1; j ++){
				#pragma ivdep
                for(int i = 1; i < NZ-1; i ++){
                    b[i][j][k] = (a[i][j][k]+a[i][j][k-1]+a[i][j][k+1]   \
                                      +a[i][j-1][k]+a[i][j+1][k]         \
                                      +a[i-1][j][k]+a[i+1][j][k])*division;
                }
            }
        }
		memcpy(a,b,NX*NY*NZ);
		//double (*tmp)[NY][NX] = a;
    		//a = b;
    		//b = tmp;
    }
}

void init(){
    a[NZ/2-1][NY/2-1][NX/2-1] = 10000;
    return;
}

void main(){
    init();
	struct timeval t1, t2;
	
	cout << "Test of Method 4" << endl;
	double timing = 0;
    for(int s = 0; s < BATCH; s ++)
    {
        gettimeofday(&t1, NULL);
        test4(100);
        gettimeofday(&t2, NULL);
		double time = 1.0*((t2).tv_sec - (t1).tv_sec) + 0.000001*((t2).tv_usec - (t1).tv_usec);
        printf("Step:%3d, Time:%.6lf\n", s, time);
		timing = timing + time/BATCH;
    }
	printf("Average Time: %.6lf\n", timing);
}
