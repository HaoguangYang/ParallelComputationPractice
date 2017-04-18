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

void test1(int S){
//Loop fission
    for(int step = 0; step < S; step ++){
        for (int i = 1; i < NZ-1; i ++){
            for(int j = 1; j < NY-1; j ++){
                for(int k = 1; k < NX-1; k ++){
                    b[i][j][k] = a[i][j][k]/7 + a[i][j][k-1]/7 + a[i][j][k+1]/7;
                }
            }
        }
		for (int i = 1; i < NZ-1; i ++){
            for(int j = 1; j < NY-1; j ++){
                for(int k = 1; k < NX-1; k ++){
					b[i][j][k] = b[i][j][k] + a[i][j-1][k]/7 + a[i][j+1][k]/7;
				}
			}
		}
		for (int i = 1; i < NZ-1; i ++){
            for(int j = 1; j < NY-1; j ++){
                for(int k = 1; k < NX-1; k ++){
					b[i][j][k] = b[i][j][k] + a[i-1][j][k]/7 + a[i+1][j][k]/7;
				}
			}
		}
		memcpy(a,b,NX*NY*NZ);
    }
}

void init(){
    a[NZ/2-1][NY/2-1][NX/2-1] = 10000;
    return;
}

void main(){
    init();
	struct timeval t1, t2;
	
	cout << "Test of Method 1" << endl;
	double timing = 0;
	for(int s = 0; s < BATCH; s ++)
    {
        gettimeofday(&t1, NULL);
        test1(100);
        gettimeofday(&t2, NULL);
		double time = 1.0*((t2).tv_sec - (t1).tv_sec) + 0.000001*((t2).tv_usec - (t1).tv_usec);
        printf("Step:%3d, Time:%.6lf\n", s, time);
		timing = timing + time/BATCH;
    }
	printf("Average Time: %.6lf\n", timing);
}
