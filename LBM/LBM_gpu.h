#ifndef LBM_GPU_H
#define LBM_GPU_H

const unsigned int scale = 2;
const unsigned int NX = 8192*scale;
const unsigned int NY = NX;
const unsigned int nThreads=32;
const unsigned int ndir = 9;

const double bytesPerMiB = 1024.0*1024.0;
const double bytesPerGiB = 1024.0*1024.0*1024.0;

const double w0 = 4.0/9.0;  // zero weight
const double ws = 1.0/9.0;  // adjacent weight
const double wd = 1.0/36.0; // diagonal weight

const double nu = 1.0/6.0;

extern "C"{
    void stream_collide_save_wrapper(double *f0, double *f1, double *f2, double *r, double *u, double *v,
         bool save, unsigned int NY1);
    void stream_collide_init(double *f0, double *f1);
    int devQuery(int maxThrdPerGPU, int deviceCount, int *maxThreadsDim, int *maxGridSize);
    int deviceSetup(const size_t &mem_size_0dir, const size_t &mem_size_n0dir, const size_t &mem_size_scalar);
    void start_mark();
    float stop_mark();
    void CUDA_finalize(void);
}

inline size_t field0_index(unsigned int x, unsigned int y)
{
    return NX*y+x;
}

inline size_t scalar_index(unsigned int x, unsigned int y)
{
    return NX*y+x;
}

inline size_t fieldn_index(unsigned int x, int y, unsigned int d)
{
    return (ndir-1)*(NX*(y+1)+x)+(d-1);
}

#endif

