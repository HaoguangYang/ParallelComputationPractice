#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <math.h>

using namespace std;

int NX=300;
int NY=300;
int NZ=300;
int BATCH=10;
const double weight = 1/7;

double*** a;
double*** b;

//double ***tmp;

void test_CacheOblivious(int S0, int S1,
                         int x0, int dx0, int x1, int dx1,
                         int y0, int dy0, int y1, int dy1,
                         int z0, int dz0, int z1, int dz1){
    //串行程序下性能改进不大，然而对并行程序可以有效减少内存读取频率，进而降低延迟。
	int dS=S1-S0;
	if (dS==1 || (x1-x0)*(y1-y0)*(z1-z0) < 1638400){
        for(int step = S0; step < S1; step ++){
            for (int x=x0+(step-S0)*dx0; x<x1+(step-S0)*dx1; x++){
                for (int y=y0+(step-S0)*dy0; y<y1+(step-S0)*dy1; y++){
                    for (int z=z0+(step-S0)*dz0; z<z1+(step-S0)*dz1; z++){
                        b[x][y][z] = (a[x][y][z]+a[x][y][z-1]+a[x][y][z+1]   \
                                      +a[x][y-1][z]+a[x][y+1][z]             \
                                      +a[x-1][y][z]+a[x+1][y][z])*weight;
                    }
                }
            }
            std::swap(a,b);
        }
    }
    else if (dS>1){
        if (2*(z1-z0)+(dz1-dz0)*dS>=4*dS){
            int z2 = (2*(z0+z1)+(dz0+dz1+2)*dS)/4;
            test_CacheOblivious(S0, S1, x0, dx0, x1, dx1, y0, dy0, y1, dy1, z0, dz0, z2, -1);
            test_CacheOblivious(S0, S1, x0, dx0, x1, dx1, y0, dy0, y1, dy1, z2, -1, z1, dz1);
    	}
    	else if (2*(y1-y0)+(dy1-dy0)*dS>=4*dS){
    	    int y2 = (2*(y0+y1)+(dy0+dy1+2)*dS)/4;
            test_CacheOblivious(S0, S1, x0, dx0, x1, dx1, y0, dy0, y2, -1, z0, dz0, z1, dz1);
            test_CacheOblivious(S0, S1, x0, dx0, x1, dx1, y2, -1, y1, dy1, z0, dz0, z1, dz1);
        }
        else{
            int S2=dS/2;
            test_CacheOblivious(S0, S0+S2, x0, dx0, x1, dx1, y0, dy0, y1, dy1, z0, dz0, z1, dz1);
            test_CacheOblivious(S0+S2, S1, x0+dx0*S2, dx0, x1+dx1*S2, dx1, y0+dy0*S2, dy0, y1+dy1*S2, dy1, z0+dz0*S2, dz0, z1+dz1*S2, dz1);
        }
    }
}


void main_loop(int x0, int x1, int y0, int y1, int z0, int z1){
    #pragma ivdep
    for (int x=x0; x<x1; x++){
        #pragma ivdep
        for (int y=y0; y<y1; y++){
            #pragma ivdep
            for (int z=z0; z<z1; z++){
                //btemp[x][y][z] = (atemp[x][y][z]+atemp[x][y][z-1]+atemp[x][y][z+1]   \
                                    +atemp[x][y-1][z]+atemp[x][y+1][z]                 \
                                    +atemp[x-1][y][z]+atemp[x+1][y][z])*weight;
                b[x][y][z] = (a[x][y][z]+a[x][y][z-1]+a[x][y][z+1]   \
                              +a[x][y-1][z]+a[x][y+1][z]             \
                              +a[x-1][y][z]+a[x+1][y][z])*weight;
            }
        }
    }
    std::swap(a,b);
}

void test_Timeskewing(int S){
    //同样没有太大改进。
    int dx0, dx1, dy0, dy1, dz0, dz1;
    int x0, x1, y0, y1, z0, z1;
    double ***atemp, ***btemp;
    for (int k0=1; k0<NZ-1; k0++){
        dz0=1;
        dz1=-1;
        //if (k0==1)
        //    dz0=0;
        if (k0==NZ-2)
            dz1=0;
        for (int j0=1; j0<NY-1; j0++){
            dy0=1;
            dy1=-1;
            //if (j0==1)
            //    dy0=0;
            if (j0==NY-2)
                dy1=0;
            for (int i0=1; i0<NX-2; i0++){
                dx0=1;
                dx1=-1;
                //if (i0==1)
                //    dx0=0;
                //atemp = a;
                //btemp = b;
                //printf("k0=%d    j0=%d    i0=%d\n", k0, j0, i0);
                for (int t=0; t<S; t++){
                    x0=max(1, i0-t*dx0);
                    y0=max(1, j0-t*dy0);
                    z0=max(1, k0-t*dz0);
                    x1=max(1, i0+1+t*dx1);
                    y1=max(1, j0+1+t*dy1);
                    z1=max(1, k0+1+t*dz1);
                    main_loop(x0,x1,y0,y1,z0,z1);
                }
            }
            //i0=NX-2
            //if (i0==NX-2)
            //    dx1=0;
            for (int t=0; t<S; t++){
                x0=max(1, NX-2-t*dx0);
                y0=max(1, j0-t*dy0);
                z0=max(1, k0-t*dz0);
                x1=i0+1;
                y1=max(1, j0+1+t*dy1);
                z1=max(1, k0+1+t*dz1);
                main_loop(x0,x1,y0,y1,z0,z1);
            }
        }
    }
}


void test_naive(int S){
    //经典循环对串行程序已经有足够好的性能表现。然而对其并行化后发现内存读取延迟造成的影响很大，24线程的实际加速比仅有2～3左右。
    //double*** A[2]={a,b};
    for(int step = 0; step < S; step ++){
        #pragma ivdep
        for (int i = 1; i < NX-1; i ++){
            #pragma ivdep
            for(int j = 1; j < NY-1; j ++){
				#pragma ivdep
                for(int k = 1; k < NZ-1; k ++){
                    b[i][j][k] = (a[i][j][k]+a[i][j][k-1]+a[i][j][k+1]   \
                                      +a[i][j-1][k]+a[i][j+1][k]         \
                                      +a[i-1][j][k]+a[i+1][j][k])*weight;
                    //A[(step+1)%2][i][j][k] = (A[step%2][i][j][k]+A[step%2][i][j][k-1]+A[step%2][i][j][k+1]   \
                                              +A[step%2][i][j-1][k]+A[step%2][i][j+1][k]                     \
                                              +A[step%2][i-1][j][k]+A[step%2][i+1][j][k])*weight;
                }
            }
        }
		//memcpy(a,b,NX*NY*NZ);
		//tmp = a;
    	//a = b;
    	//b = tmp;
    	std::swap(a,b);
    }
}

void init(){
    a = (double***)malloc(NX*sizeof(double**));
    b = (double***)malloc(NX*sizeof(double**));
    for (int i=0; i<NX; i++){
        a[i] = (double**)malloc(NY*sizeof(double*));
        b[i] = (double**)malloc(NY*sizeof(double*));
        for (int j=0; j<NY; j++){
            a[i][j] = (double*)malloc(NZ*sizeof(double));
            b[i][j] = (double*)malloc(NZ*sizeof(double));
        }
    }
    a[NZ/2-1][NY/2-1][NX/2-1] = 10000;
    return;
}

void main(int argc, char* argv[]){
    int loop;
    if (argc == 6){
        NX=atoi(argv[1]);
        NY=atoi(argv[2]);
        NZ=atoi(argv[3]);
        BATCH=atoi(argv[4]);
        loop = atoi(argv[5]);
    }
    else{
        cout << "Usage: test 100 200 300 10 100 to set size to 100*200*300 and iterate 10 batches each looping 100 steps." << endl;
        return;
    }
    init();
	struct timeval t1, t2;
	
	cout << "Test of Method 4" << endl;
	double timing = 0;
    for(int s = 0; s < BATCH; s ++)
    {
        gettimeofday(&t1, NULL);
        //test_naive(loop);
        test_Timeskewing(loop);
        //test_CacheOblivious(s*loop, (s+1)*loop, 1, 0, NX-1, 0, 1, 0, NY-1, 0, 1, 0, NZ-1, 0);
        gettimeofday(&t2, NULL);
		double time = 1.0*((t2).tv_sec - (t1).tv_sec) + 0.000001*((t2).tv_usec - (t1).tv_usec);
        printf("Batch:%3d, Time:%.6lf\n", s, time);
		timing = timing + time/BATCH;
    }
	printf("Average Time: %.6lf\n", timing);
	for (int i=0; i<NX; i++){
	    for (int j=0; j<NY; j++){
	        free(a[i][j]);
	        free(b[i][j]);
	    }
	}
	for (int i=0; i<NX; i++){
	    free (a[i]);
	    free (b[i]);
	}
	free(a);
	free(b);
	return;
}
