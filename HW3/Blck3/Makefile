# Generated automatically from Makefile.in by configure.
ALL: jacobi
SHELL = /bin/sh
CFLAGS = -std=c++11 -O3 -xAVX -openmp -DMKL_VML_MIXED
CLIBS = -openmp
DIRS = 
jacobi: jacobi.cpp
	$(SHELL) /opt/intel/bin/compilervars.sh intel64
	mpiicc $(CFLAGS) -o jacobi jacobi.cpp wxl.a -lm -lstdc++ -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5
profile.alog: jacobi.cpp
	mpiicc $(CFLAGS) -o jacobi.log -mpilog jacobi.cpp -lm -lstdc++
	mpirun -np 4 jacobi.log
	mv jacobi.log_profile.log profile.alog
clean:
	rm -f jacobi jacobi.o jacobi.log
	#for dir in $(DIRS) ; do \
	#    ( cd $$dir ; make clean ) ; done
