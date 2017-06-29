ALL: LBM_hybrid
SHELL = /bin/sh
CFLAGS = -std=c++11 -O3 -xAVX -openmp -DMKL_VML_MIXED
CLIBS = -openmp
CUDA_PATH ?= "/usr/local/cuda"
LINKERFLAGS = -lm -lstdc++ -liomp5 -L$(CUDA_PATH)/lib64 -lcudart

clean:
	rm -f LBM *.o *.bin

# change sm_20 to match the architecture of your device
# allowed values are sm_Nn
# where N is the major version number and n is the minor number
# 'sm_20','sm_21','sm_30','sm_32','sm_35','sm_37','sm_50','sm_52','sm_53'
LBM.o: LBM_gpu.cu LBM.cpp
	nvcc -arch sm_35 -v --ptxas-options=-v -O3 -c LBM_gpu.cu
	mpiicc $(CFLAGS) -c LBM.cpp

main.o: main.cpp
	$(SHELL) /opt/intel/bin/compilervars.sh intel64 
	mpiicc $(CFLAGS) -c main.cpp seconds.cpp

LBM_hybrid: main.o LBM.o
	mpiicc $(CFLAGS) -o LBM LBM.o main.o LBM_gpu.o seconds.o $(LINKERFLAGS) 

