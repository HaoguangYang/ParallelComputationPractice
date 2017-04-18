default:HW2

CC = icc
CFLAGS = -O3 -xAVX -qopt-report-phase=vec -openmp
CLIBS = -openmp

HW2:pi_omp.o 
	$(CC) $(CLIBS) pi_omp.o -o HW2

pi_omp.o:pi_omp.c
	$(CC) $(CFLAGS) -c pi_omp.c -o pi_omp.o

serial:pi.o
	$(CC) $(CFLAGS) pi.o -o pi

pi.o:pi_serial.c
	$(CC) $(CFLAGS) -c pi_serial.c -o pi.o

clean:
	rm -rf ./*.o ./*.optrpt HW2 pi

run:
	./HW2

run_serial:
	./pi
