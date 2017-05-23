# Generated automatically from Makefile.in by configure.
ALL: jacobi
SHELL = /bin/sh
CFLAGS = -O3 -xAVX -openmp
CLIBS = -openmp
DIRS = 
jacobi: jacobi.cpp
	mpiicc $(CFLAGS) -o jacobi jacobi.cpp wxl.a -lm -lstdc++
profile.alog: jacobi.cpp
	mpiicc $(CFLAGS) -o jacobi.log -mpilog jacobi.cpp -lm -lstdc++
	mpirun -np 4 jacobi.log
	mv jacobi.log_profile.log profile.alog
clean:
	rm -f jacobi jacobi.o jacobi.log
	#for dir in $(DIRS) ; do \
	#    ( cd $$dir ; make clean ) ; done
