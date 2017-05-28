#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#define TIME(a,b) (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))


/* This example handles a 12 x 12 mesh, on 4 processors only. */

int maxm;
int maxn;
int maxp;
int BATCH;

int MPI_GS1;
int MPI_GS2;
int MPI_GS3;
int rank;
int size;

#ifdef __cplusplus
extern "C" {
#endif

//extern int Init(double *data, long long L);
//extern int Check(double *data, long long L);

#ifdef __cplusplus
}
#endif

void InitVal(double ***xlocal, int i1, int j1, int k1, int ind1, int ind2, int ind3){
    for (int i=0; i<=i1; i++) 
	for (int j=0; j<=j1; j++) 
	for (int k=0; k<=k1; k++)
	    xlocal[i][j][k] = rank;

    if (ind1 == 0){
		for (int j=0; j<=j1; j++)
		for (int k=0; k<=k1; k++)
			xlocal[0][j][k]=-1.0;
	}
	//if (rank==0) printf( "Boundary set on side 1...\n" );

	if (ind2 == 0){
		for (int i=0; i<=i1; i++)
		for (int k=0; k<=k1; k++)
			xlocal[i][0][k]=-1.0;
	}
	//if (rank==0) printf( "Boundary set on side 2...\n" );
	
	if (ind3 == 0){
		for (int i=0; i<=i1; i++)
		for (int j=0; j<=j1; j++)
			xlocal[i][j][0]=-1.0;
	}
	//if (rank==0) printf( "Boundary set on side 3...\n" );
	
	if (ind1 == MPI_GS1-1){
		for (int j=0; j<=j1; j++)
		for (int k=0; k<=k1; k++)
			xlocal[i1][j][k]=-1.0;
	}
	//if (rank==0) printf( "Boundary set on side 4...\n" );

	if (ind2 == MPI_GS2-1){
		for (int i=0; i<=i1; i++)
		for (int k=0; k<=k1; k++)
			xlocal[i][j1][k]=-1.0;
	}
	//if (rank==0) printf( "Boundary set on side 5...\n" );
	
	if (ind3 == MPI_GS3-1){
		for (int i=0; i<=i1; i++)
		for (int j=0; j<=j1; j++)
			xlocal[i][j][k1]=-1.0;	//Set Boundary
	}
	//if (rank==0) printf( "Boundary set on side 6...\n" );
	return;
}

int main(int argc, char* argv[]) {
    int        value, errcnt, toterr, i, j, k, loop, t;
    int        i0, i1, j0, j1, k0, k1;
    MPI_Status status;
    double     diffnorm, gdiffnorm;
    double     ***xlocal;
    double     ***xnew;
	struct timeval t1, t2;
    
    if (argc == 5){
        maxm=atoi(argv[1]);
        maxn=atoi(argv[2]);
        maxp=atoi(argv[3]);
        loop = atoi(argv[4]);       //itcnt
    }
    else{
        printf("Usage: test 100 200 300 10 to set size to 100*200*300 and looping 10 time steps.\n");
        return -1;
    }
    
    MPI_Init( &argc, &argv );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    if (rank==0) printf("Size:%dx%dx%d, # of Steps: %d, # of procs: %d, # of threads per proc: %d\n", \
        maxm, maxn, maxp, loop, size, omp_get_num_threads());

    /* xlocal[][0] is lower ghostpoints, xlocal[][maxn+2] is upper */

    /* Note that top and bottom processes have one less row of interior
       points */
       

    //**************************************************************************
    //Assigning Works and Senmentation of Global Mesh***************************
    //**************************************************************************
    
	double unit_length = pow(maxm*maxn*maxp/(double)size, 1.0/3.0);	//The block allocated to each mpi node
	MPI_GS1 = floor(maxm/unit_length);
	MPI_GS2 = floor(maxn/unit_length);
	MPI_GS3 = floor(maxp/unit_length);
	if (MPI_GS1 == 0){
		MPI_GS1 = 1;
		if (MPI_GS2 == 0){
			MPI_GS2 = 1;
			unit_length = maxp/(double)size;
			MPI_GS3 = floor(maxp/unit_length);
			if (MPI_GS3 == 0){
				MPI_GS3 = 1;
			}
		}
		else if (MPI_GS3 == 0){
			MPI_GS3 = 1;
			unit_length = maxn/(double)size;
			MPI_GS2 = std::max(floor(maxn/unit_length),1.0);
		}
		else{
			unit_length = pow(maxn*maxp/(double)size, 1.0/2.0);
			MPI_GS2 = std::max(floor(maxn/unit_length),1.0);
			MPI_GS3 =std::max(floor(maxp/unit_length),1.0);
		}
	}
	else if (MPI_GS2 == 0){
		MPI_GS2 = 1;
		if (MPI_GS3 == 0){
			MPI_GS3 = 1;
			unit_length = maxm/(double)size;
			MPI_GS1 = std::max(floor(maxm/unit_length),1.0);
		}
		else{
			unit_length = pow(maxm*maxp/(double)size, 1.0/2.0);
			MPI_GS1 = std::max(floor(maxm/unit_length),1.0);
			MPI_GS3 = std::max(floor(maxp/unit_length),1.0);
		}
	}
	else if (MPI_GS3 == 0){
		MPI_GS3 = 1;
		unit_length = pow(maxm*maxn/(double)size, 1.0/2.0);
		MPI_GS1 = std::max(floor(maxm/unit_length),1.0);
		MPI_GS2 = std::max(floor(maxn/unit_length),1.0);
	}
	
	int MPI_Grid1[MPI_GS1+1];
	int MPI_Grid2[MPI_GS2+1];
	int MPI_Grid3[MPI_GS3+1];

	if (MPI_GS1*MPI_GS2*MPI_GS3<size) {
		printf("Improper Cluster Size! Please Utilize More Nodes.\n");
		return -1;
	}
	for (int i=0; i<std::max(std::max(MPI_GS1,MPI_GS2),MPI_GS3); i++){
		int tmp = unit_length*i;
		if (i<MPI_GS1)	MPI_Grid1[i] = tmp;
		if (i<MPI_GS2)	MPI_Grid2[i] = tmp;
		if (i<MPI_GS3)	MPI_Grid3[i] = tmp;
		//printf( "%d %d %d %d %d\n", tmp, i, MPI_Grid1[i], MPI_Grid2[i], MPI_Grid3[i]);
	}
	MPI_Grid1[MPI_GS1] = maxm;
	MPI_Grid2[MPI_GS2] = maxn;
	MPI_Grid3[MPI_GS3] = maxp;
	//printf( "%d %d %d\n", MPI_Grid3[0], MPI_Grid3[1], MPI_Grid3[2]);
	
	int ind1 = rank%MPI_GS1;
	int ind2 = (rank/MPI_GS1)%MPI_GS2;
	int ind3 = rank/(MPI_GS1*MPI_GS2);
	
	i0 = 1;
	i1 = MPI_Grid1[ind1+1]-MPI_Grid1[ind1];
	j0 = 1;
	j1 = MPI_Grid2[ind2+1]-MPI_Grid2[ind2];
	k0 = 1;
	k1 = MPI_Grid3[ind3+1]-MPI_Grid3[ind3];
	int xsize[3] = {i1+2, j1+2, k1+2};
	
	//printf( "%d %d %d %d %f %d %d\n", ind3, xsize[0], xsize[1], xsize[2], unit_length, MPI_Grid1[0], MPI_Grid1[1]);
	
	double *data1 = (double *)malloc((i1+2)*(j1+2)*(k1+2)*sizeof(double));
	xlocal = (double***)malloc((i1+2)*sizeof(double**));
    for (int i=0; i<i1+2; i++){
		xlocal[i] = (double**)malloc((j1+2)*sizeof(double*));
        for (int j=0; j<j1+2; j++){
            //xlocal[i][j] = (double*)malloc((k1+2)*sizeof(double));
            xlocal[i][j] = &(data1[i*(j1+2)*(k1+2)+j*(k1+2)]);
            //xnew[i][j] = (double*)malloc((k1+2)*sizeof(double));
			//printf("%d    %d\n",i,j);
        }
    }

	if (rank == 0) printf( "Work Assignment Done...\n\
		i0=%d    i1=%d    j0=%d    j1=%d    k0=%d    k1=%d\n\
		GS1=%d   GS2=%d   GS3=%d\n\
		ind1=%d  ind2=%d  ind3=%d\n",\
		i0, i1, j0, j1, k0, k1, MPI_GS1, MPI_GS2, MPI_GS3, ind1, ind2, ind3 );

	/* Fill the data as specified */
    //InitVal(xlocal, i1, j1, k1, ind1, ind2, ind3);
    //Init(&xlocal[0][0][0], i1*j1*k1);
	double sum = 0.0;
	for (i=i0; i<=i1; i++) {
		#pragma omp parallel for schedule(guided)
	    for (j=j0; j<=j1; j++) {
			gettimeofday(&t1, NULL);
			srand((unsigned)t1.tv_usec);
			for (k=k0; k<=k1; k++) {
				xlocal[i][j][k] = rand()/(double)RAND_MAX-0.500;
				sum += xlocal[i][j][k];
			}
		}
	}
	if (rank==0) printf( "Boundaries are set ...\n" );
	
	if (ind1 == 0)			i0++;
    if (ind1 == MPI_GS1-1) 	i1--;
	if (ind2 == 0)        	j0++;
    if (ind2 == MPI_GS2-1) 	j1--;
	if (ind3 == 0)        	k0++;
    if (ind3 == MPI_GS3-1) 	k1--;	//Filter out boundary
	
	double *data2 = (double *)malloc((i1+1-i0)*(j1+1-j0)*(k1+1-k0)*sizeof(double));
	xnew = (double***)malloc((i1+1-i0)*sizeof(double**));
	for (int i=0; i<i1+1-i0; i++){
		xnew[i] = (double**)malloc((j1+1-j0)*sizeof(double*));
		for (int j=0; j<j1+1-j0; j++){
			xnew[i][j] = &(data2[i*(j1+1-j0)*(k1+1-k0)+j*(k1+1-k0)]);
		}
	}
	//Init(&xnew[0][0][0], (i1+1-i0)*(j1+1-j0)*(k1+1-k0));
	
    if (rank == 0) printf( "Initialization Complete...\n" );
	//printf( "Initial Sum = %f\n", sum );
	
    MPI_Barrier(MPI_COMM_WORLD); gettimeofday(&t1, NULL);
    
    
    //**************************************************************************
    //Main Loop*****************************************************************
    //**************************************************************************
    
    for (t=0; t<loop; t++){
	/* Send up unless I'm at the top, then receive from below */
	/* Note the use of xlocal[i] for &xlocal[i][0] */
	MPI_Datatype slicei;
	int subsizei[3] = {1, xsize[1]-2, xsize[2]-2};
	
	if (ind1 < MPI_GS1 - 1){
	    //MPI_Send( &(xlocal[i1]), (j1+2)*(k1+2), MPI_DOUBLE, rank + 1, 0, 
		//      MPI_COMM_WORLD );
		int starts[3] = {i1,1,1};
		MPI_Type_create_subarray(3, xsize, subsizei, starts, MPI_ORDER_C, MPI_DOUBLE, &slicei);
		MPI_Type_commit(&slicei);
	    MPI_Send( &xlocal[0][0][0], 1, slicei, rank + 1, 0, 
		      MPI_COMM_WORLD );
		//printf( "Flag1.1\n" );
	}
	if (ind1 > 0){
	    //MPI_Recv( &(xlocal[0]), (j1+2)*(k1+2), MPI_DOUBLE, rank - 1, 0, 
		//      MPI_COMM_WORLD, &status );
		int starts[3] = {0,1,1};
		MPI_Type_create_subarray(3, xsize, subsizei, starts, MPI_ORDER_C, MPI_DOUBLE, &slicei);
		MPI_Type_commit(&slicei);
	    MPI_Recv( &xlocal[0][0][0], 1, slicei, rank - 1, 0, 
		      MPI_COMM_WORLD, &status );
		//printf( "Flag1.2\n" );
	}
	/* Send down unless I'm at the bottom */
	if (ind1 > 0){
	    //MPI_Send( &(xlocal[1]), (j1+2)*(k1+2), MPI_DOUBLE, rank - 1, 1, 
		//      MPI_COMM_WORLD );
		int starts[3] = {1,1,1};
		MPI_Type_create_subarray(3, xsize, subsizei, starts, MPI_ORDER_C, MPI_DOUBLE, &slicei);
		MPI_Type_commit(&slicei);
	    MPI_Send( &xlocal[0][0][0], 1, slicei, rank - 1, 1, 
		      MPI_COMM_WORLD );
		//printf( "Flag1.3\n" );
	}
	if (ind1 < MPI_GS1 - 1){
	    //MPI_Recv( &(xlocal[i1+1]), (j1+2)*(k1+2), MPI_DOUBLE, rank + 1, 1, 
		//      MPI_COMM_WORLD, &status );
		int starts[3] = {i1+1,1,1};
		MPI_Type_create_subarray(3, xsize, subsizei, starts, MPI_ORDER_C, MPI_DOUBLE, &slicei);
		MPI_Type_commit(&slicei);
	    MPI_Recv( &xlocal[0][0][0], 1, slicei, rank + 1, 1, 
		      MPI_COMM_WORLD, &status );
		//printf( "Flag1.4\n" );
	}
	//printf( "Communication of Dim1 Done\n" );
	//Dim 2
	//double toSend2[i1][k1];
	//double toRecv2[i1][k1];
	MPI_Datatype slicej;
	int subsizej[3] = {xsize[0]-2, 1, xsize[2]-2};
	//MPI_Type_vector(k1, 1, (k1+2)*(j1+2), MPI_DOUBLE, &slice1);
	//MPI_Type_commit (&slice1);
	//MPI_Type_create_resized (slice1, 0, 1*sizeof(double), &slicej);
	//Usage: MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &newtype);
	//MPI_Type_create_subarray(3, xsize, subsizej, starts, MPI_ORDER_C, MPI_DOUBLE, &slicej);
	//MPI_Type_commit(&slicej);
	//MPI_Barrier(MPI_COMM_WORLD);
	if (ind2 < MPI_GS2 - 1){
		int starts[3] = {1,j1,1};
		MPI_Type_create_subarray(3, xsize, subsizej, starts, MPI_ORDER_C, MPI_DOUBLE, &slicej);
		MPI_Type_commit(&slicej);
	    MPI_Send( &xlocal[0][0][0], 1, slicej, rank + MPI_GS1, 2, 
		      MPI_COMM_WORLD );
		//printf( "Flag2.1\n" );
	}
	if (ind2 > 0){
		int starts[3] = {1,0,1};
		MPI_Type_create_subarray(3, xsize, subsizej, starts, MPI_ORDER_C, MPI_DOUBLE, &slicej);
		MPI_Type_commit(&slicej);
	    MPI_Recv( &xlocal[0][0][0], 1, slicej, rank - MPI_GS1, 2, 
		      MPI_COMM_WORLD, &status );
		//printf( "Flag2.2\n" );
	}
	/* Send down unless I'm at the bottom */
	if (ind2 > 0){
		int starts[3] = {1,1,1};
		MPI_Type_create_subarray(3, xsize, subsizej, starts, MPI_ORDER_C, MPI_DOUBLE, &slicej);
		MPI_Type_commit(&slicej);
	    MPI_Send( &xlocal[0][0][0], 1, slicej, rank - MPI_GS1, 3, 
		      MPI_COMM_WORLD );
		//printf( "Flag2.3\n" );
	}
	if (ind2 < MPI_GS2 - 1){
		int starts[3] = {1,j1+1,1};
		MPI_Type_create_subarray(3, xsize, subsizej, starts, MPI_ORDER_C, MPI_DOUBLE, &slicej);
		MPI_Type_commit(&slicej);
	    MPI_Recv( &xlocal[0][0][0], 1, slicej, rank + MPI_GS1, 3, 
		      MPI_COMM_WORLD, &status );
		//printf( "Flag2.4\n" );
	}
	//printf( "Communication of Dim2 Done\n" );
	//Dim3
	//double toSend3[i1][j1];
	//double toRecv3[i1][j1];
	MPI_Datatype slicek;
	int subsizek[3] = {xsize[0]-2, xsize[1]-2, 1};
	///int starts[3] = {0,0,0};
	//MPI_Type_vector(1, 1, k1+2, MPI_DOUBLE, &slice2);
	//MPI_Type_commit (&slice2);
	//MPI_Type_create_resized (slice2, 0, 1*sizeof(double), &slicek);
	//MPI_Type_create_subarray(3, xsize, subsizek, starts, MPI_ORDER_C, MPI_DOUBLE, &slicek);
	//MPI_Type_commit(&slicek);
	//MPI_Barrier(MPI_COMM_WORLD);
	if (ind3 < MPI_GS3 - 1){
		int starts[3] = {1,1,k1};
		MPI_Type_create_subarray(3, xsize, subsizek, starts, MPI_ORDER_C, MPI_DOUBLE, &slicek);
		MPI_Type_commit(&slicek);
	    MPI_Send( &xlocal[0][0][0], 1, slicek, rank + MPI_GS1*MPI_GS2, 4, 
		      MPI_COMM_WORLD );
	}
	if (ind3 > 0){
		int starts[3] = {1,1,0};
		MPI_Type_create_subarray(3, xsize, subsizek, starts, MPI_ORDER_C, MPI_DOUBLE, &slicek);
		MPI_Type_commit(&slicek);
	    MPI_Recv( &xlocal[0][0][0], 1, slicek, rank - MPI_GS1*MPI_GS2, 4, 
		      MPI_COMM_WORLD, &status );
	}
	/* Send down unless I'm at the bottom */
	if (ind3 > 0){
		int starts[3] = {1,1,1};
		MPI_Type_create_subarray(3, xsize, subsizek, starts, MPI_ORDER_C, MPI_DOUBLE, &slicek);
		MPI_Type_commit(&slicek);
	    MPI_Send( &xlocal[0][0][0], 1, slicek, rank - MPI_GS1*MPI_GS2, 5, 
		      MPI_COMM_WORLD );
	}
	if (ind3 < MPI_GS3 - 1){
		int starts[3] = {1,1,k1+1};
		MPI_Type_create_subarray(3, xsize, subsizek, starts, MPI_ORDER_C, MPI_DOUBLE, &slicek);
		MPI_Type_commit(&slicek);
	    MPI_Recv( &xlocal[0][0][0], 1, slicek, rank + MPI_GS1*MPI_GS2, 5, 
		      MPI_COMM_WORLD, &status );
	}
	//printf( "Communication of Dim3 Done\n" );
	/* Compute new values (but not on boundary) */
	if (rank==0) printf( "Communication of timestep %d is done...\n", t );
	//diffnorm = 0.0;
	//MPI_Barrier(MPI_COMM_WORLD);                          //All Data is Updated
	#pragma omp parallel for schedule(guided)
	#pragma ivdep
	for (i=i0; i<=i1; i++) {
	    #pragma ivdep
	    for (j=j0; j<=j1; j++) {
			#pragma ivdep
			for (k=k0; k<=k1; k++) {
				xnew[i-i0][j-j0][k-k0] = 0.1*xlocal[i][j+1][k] + 0.1*xlocal[i][j-1][k] + \
										 0.1*xlocal[i+1][j][k] + 0.1*xlocal[i-1][j][k] + \
										 0.1*xlocal[i][j][k+1] + 0.1*xlocal[i][j][k-1] + 0.4*xlocal[i][j][k];
				//diffnorm += (xnew[i][j][k] - xlocal[i][j][k]) * \
							(xnew[i][j][k] - xlocal[i][j][k]);
			}
	    }
	}
	/* Only transfer the interior points */
	#pragma omp parallel for schedule(guided)
	for (i=i0; i<=i1; i++) {
	    for (j=j0; j<=j1; j++) {
			for (k=k0; k<=k1; k++) {
				xlocal[i][j][k] = xnew[i-i0][j-j0][k-k0];
			}
		}
	}
	//MPI_Allreduce( &diffnorm, &gdiffnorm, 1, MPI_DOUBLE, MPI_SUM, \
		       MPI_COMM_WORLD );
	//gdiffnorm = sqrt( gdiffnorm );
	//if (rank == 0) printf( "At iteration %d, diff is %e\n", loop, \
			       gdiffnorm );
    }   // && gdiffnorm > 1.0e-2
    
    MPI_Barrier(MPI_COMM_WORLD); gettimeofday(&t2, NULL);
    
    if (rank==0) printf("Completed %d time steps in %f secs.\n", loop, TIME(t1,t2));
    
	//Check(&xnew[0][0][0], (xsize[0]-2)*(xsize[1]-2)*(xsize[2]-2));
	double sum2 = 0.0;
	for (i=1; i<=xsize[0]-2; i++) {
	    for (j=1; j<=xsize[1]-2; j++) {
			for (k=1; k<=xsize[2]-2; k++) {
				sum2+=xlocal[i][j][k];
			}
		}
	}
	//printf( "Final Sum = %f\n", sum2 );
	double surface = 2*(maxm*maxn+maxn*maxp+maxm*maxp);
	double diff = fabs(sum-sum2)/surface/loop;
	//printf( "Average Difference = %f\n", diff );
	if (diff < 0.001 && surface>1000) printf( "Result is Correct.\n" );
	else if (surface<=1000) printf("Mesh size is too small for randomness to damp down.\n");
	else printf("Result is Wrong.\n");
	//Don't forget to free array!!!
	free(xlocal);
	free(xnew);
	MPI_Finalize();
    return 0;
}
