#include <stdio.h>
#include <stdlib.h>
//#include "GPU_utils.h"
#include "LBM_gpu.h"
#include <cuda.h>

double *f0_gpu,*f1_gpu,*f2_gpu;
double *rho_gpu,*ux_gpu,*uy_gpu;
double size_field_data;
double size_scalar_data;
//double *swapy0_gpu, *swapy1_gpu;

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y)
{
    return NX*y+x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, int y, unsigned int d)
{
    return (ndir-1)*(NX*(y+1)+x)+(d-1);
}

#define checkCudaErrors(err)  __checkCudaErrors(err,#err,__FILE__,__LINE__)
#define getLastCudaError(msg)  __getLastCudaError(msg,__FILE__,__LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line )
{
    if(err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d)\"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gpu_stream_collide_save(double *f0, double *f1, double *f2, double *r, double *u, double *v, bool save, unsigned int ymax)
{
    // useful constants
    const double tauinv = 2.0/(6.0*nu+1.0); // 1/tau
    const double omtauinv = 1.0-tauinv;     // 1 - 1/tau

    unsigned int y = blockIdx.y+1;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
    //if (y>=1 && y<NY1-1){
    unsigned int xp1 = (x+1)%NX;
    unsigned int yp1 = y+1;
    unsigned int xm1 = (NX+x-1)%NX;
    unsigned int ym1 = y-1;
    
    /*if (!last_save){
        if (y > 0 && y < ndir && x<NX){
            f2[gpu_fieldn_index(x,0,y)] = swapy0[gpu_fieldn_index(x,-1,y)];
            f2[gpu_fieldn_index(x,ymax+1,y)] = swapy1[gpu_fieldn_index(x,-1,y)];
        }
        f0[gpu_scalar_index(x,0)] = swapy0[gpu_fieldn_index(0,0,x+1)];
        f0[gpu_scalar_index(x,ymax+1)] = swapy1[gpu_fieldn_index(0,0,x+1)];
    }
    __syncthreads();*/
    // direction numbering scheme
    // 6 2 5
    // 3 0 1
    // 7 4 8
    
    double ft0 = f0[gpu_scalar_index(x,y)];
    
    // load populations from adjacent nodes
    double ft1 = f1[gpu_fieldn_index(xm1,y,  1)];
    double ft2 = f1[gpu_fieldn_index(x,  ym1,2)];
    double ft3 = f1[gpu_fieldn_index(xp1,y,  3)];
    double ft4 = f1[gpu_fieldn_index(x,  yp1,4)];
    double ft5 = f1[gpu_fieldn_index(xm1,ym1,5)];
    double ft6 = f1[gpu_fieldn_index(xp1,ym1,6)];
    double ft7 = f1[gpu_fieldn_index(xp1,yp1,7)];
    double ft8 = f1[gpu_fieldn_index(xm1,yp1,8)];
    
    // compute moments
    double rho = ft0+ft1+ft2+ft3+ft4+ft5+ft6+ft7+ft8;
    double rhoinv = 1.0/rho;
    
    double ux = rhoinv*(ft1+ft5+ft8-(ft3+ft6+ft7));
    double uy = rhoinv*(ft2+ft5+ft6-(ft4+ft7+ft8));
    
    if(save)
    {
        r[gpu_scalar_index(x,y)] = rho;
        u[gpu_scalar_index(x,y)] = ux;
        v[gpu_scalar_index(x,y)] = uy;
    }
    
    // now compute and relax to equilibrium
    // note that
    // relax to equilibrium
    // feq_i  = w_i rho [1 + 3(ci . u) + (9/2) (ci . u)^2 - (3/2) (u.u)]
    // feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u) + (1/2) (ci . 3u)^2]
    // feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u){ 1 + (1/2) (ci . 3u) }]
    
    // temporary variables
    double tw0r = tauinv*w0*rho; //   w[0]*rho/tau 
    double twsr = tauinv*ws*rho; // w[1-4]*rho/tau
    double twdr = tauinv*wd*rho; // w[5-8]*rho/tau
    double omusq = 1.0 - 1.5*(ux*ux+uy*uy); // 1-(3/2)u.u
    
    double tux = 3.0*ux;
    double tuy = 3.0*uy;
    
    f0[gpu_scalar_index(x,y)]    = omtauinv*ft0  + tw0r*(omusq);
    
    double cidot3u = tux;
    f2[gpu_fieldn_index(x,y,1)]  = omtauinv*ft1  + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuy;
    f2[gpu_fieldn_index(x,y,2)]  = omtauinv*ft2  + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux;
    f2[gpu_fieldn_index(x,y,3)]  = omtauinv*ft3  + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy;
    f2[gpu_fieldn_index(x,y,4)]  = omtauinv*ft4  + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = tux+tuy;
    f2[gpu_fieldn_index(x,y,5)]  = omtauinv*ft5  + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuy-tux;
    f2[gpu_fieldn_index(x,y,6)]  = omtauinv*ft6  + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -(tux+tuy);
    f2[gpu_fieldn_index(x,y,7)]  = omtauinv*ft7  + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tux-tuy;
    f2[gpu_fieldn_index(x,y,8)]  = omtauinv*ft8  + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
}

__host__ void stream_collide_save_wrapper(double *f0, double *f1, double *f2, double *r, double *u, double *v,
         bool save, unsigned int NY1)
{
    //checkCudaErrors(cudaMemcpy(&f0_gpu[0], &f0[0], NX*sizeof(double), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(&f0_gpu[NX*(NY1-1)], &f0[NX*(NY1-1)], NX*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&f1_gpu[NX*(ndir-1)], &f1[NX*(ndir-1)], NX*(ndir-1)*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(f1_gpu[NX*(ndir-1)*NY1]), &(f1[NX*(ndir-1)*NY1]), NX*(ndir-1)*sizeof(double), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(f1_gpu, f1, size_field_data, cudaMemcpyHostToDevice));
    // blocks in grid
    dim3  grid(NX/nThreads, NY1-2, 1);
    // threads in block
    dim3  threads(nThreads, 1, 1);
    gpu_stream_collide_save<<< grid, threads >>>(f0_gpu,f1_gpu,f2_gpu,rho_gpu,ux_gpu,uy_gpu,save, NY1-1);
    
    if(save) {
        checkCudaErrors(cudaMemcpy(f0, f0_gpu, size_scalar_data, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(f2, f2_gpu, size_field_data, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(r, rho_gpu, size_scalar_data, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(u, ux_gpu, size_scalar_data, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(v, uy_gpu, size_scalar_data, cudaMemcpyDeviceToHost));
    }
    else {
        //checkCudaErrors(cudaMemcpy(swapy0, swapy0_gpu, NX*sizeof(double)*ndir, cudaMemcpyDeviceToHost));
        //checkCudaErrors(cudaMemcpy(swapy1, swapy1_gpu, NX*sizeof(double)*ndir, cudaMemcpyDeviceToHost));
        //checkCudaErrors(cudaMemcpy(&f0[NX], &f0_gpu[NX], NX*sizeof(double), cudaMemcpyDeviceToHost));
        //checkCudaErrors(cudaMemcpy(&f0[NX*(NY1-2)], &f0_gpu[NX*(NY1-2)], NX*sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&f2[2*NX*(ndir-1)], &f2_gpu[2*NX*(ndir-1)], NX*sizeof(double)*(ndir-1), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&f2[NX*(ndir-1)*(NY1-1)], &f2_gpu[NX*(ndir-1)*(NY1-1)], NX*sizeof(double)*(ndir-1), cudaMemcpyDeviceToHost));
        //checkCudaErrors(cudaMemcpy(f2, f2_gpu, size_field_data, cudaMemcpyDeviceToHost));
    }
    
    double *temp = f1_gpu;
    f1_gpu = f2_gpu;
    f2_gpu = temp;
    
    getLastCudaError("gpu_stream_collide_save kernel error");
}

__host__ void stream_collide_init(double *f0, double *f1)
{
    checkCudaErrors(cudaMemcpy(f0_gpu, f0, size_scalar_data, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(f1_gpu, f1, size_field_data, cudaMemcpyHostToDevice));
}

cudaEvent_t start, stop;

int devQuery(int maxThrdPerGPU, int deviceCount, int *maxThreadsDim, int *maxGridSize){
    checkCudaErrors(cudaSetDevice(0));
    int dev;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        size_t gpu_free_mem, gpu_total_mem;
        checkCudaErrors(cudaMemGetInfo(&gpu_free_mem,&gpu_total_mem));
        if (dev == 0) printf("=================CUDA information=================\n");
        printf("             device: %d\n", dev);
        printf("               name: %s\n",deviceProp.name);
        printf("    multiprocessors: %d\n",deviceProp.multiProcessorCount);
        printf(" compute capability: %d.%d\n",deviceProp.major,deviceProp.minor);
        printf("      global memory: %.1f MiB\n",deviceProp.totalGlobalMem/bytesPerMiB);
        printf("        free memory: %.1f MiB\n",gpu_free_mem/bytesPerMiB);
        printf("\n");
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("**************************************************\n");
        if (dev == 0){
            maxThrdPerGPU = deviceProp.multiProcessorCount*deviceProp.maxThreadsPerMultiProcessor;
            maxThreadsDim = deviceProp.maxThreadsDim;
            maxGridSize = deviceProp.maxGridSize;
        }
    }
    return 0;
}


int deviceSetup(const size_t &mem_size_0dir, const size_t &mem_size_n0dir, const size_t &mem_size_scalar){
    int dev = 0;
    cudaSetDevice(dev);
    
    checkCudaErrors(cudaMalloc((void**)&f0_gpu,mem_size_0dir));
    checkCudaErrors(cudaMalloc((void**)&f1_gpu,mem_size_n0dir));
    checkCudaErrors(cudaMalloc((void**)&f2_gpu,mem_size_n0dir));
    checkCudaErrors(cudaMalloc((void**)&rho_gpu,mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&ux_gpu,mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&uy_gpu,mem_size_scalar));
    //checkCudaErrors(cudaMalloc((void**)&swapy0_gpu,NX*sizeof(double)*ndir));
    //checkCudaErrors(cudaMalloc((void**)&swapy1_gpu,NX*sizeof(double)*ndir));
    size_field_data = mem_size_n0dir;
    size_scalar_data = mem_size_scalar;
    // create event objects
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    return 0;
}

void start_mark(){
    checkCudaErrors(cudaEventRecord(start,0));
}

float stop_mark(){
    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds,start,stop));
    return milliseconds;
}

void CUDA_finalize(){
    // destory event objects
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    
    // free all memory allocatd on the GPU and host
    checkCudaErrors(cudaFree(f0_gpu));
    checkCudaErrors(cudaFree(f1_gpu));
    checkCudaErrors(cudaFree(f2_gpu));
    checkCudaErrors(cudaFree(rho_gpu));
    checkCudaErrors(cudaFree(ux_gpu));
    checkCudaErrors(cudaFree(uy_gpu));
    
    // release resources associated with the GPU device
    cudaDeviceReset();
    return;
}
