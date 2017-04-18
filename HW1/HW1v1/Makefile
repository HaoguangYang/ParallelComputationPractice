default:HW1

CC = icc
CFLAGS = -O3 -xAVX -qopt-report-phase=vec -openmp
CLIBS = -openmp

HW1:main.o 
	$(CC) $(CLIBS) main.o -o HW1

main.o:main.cpp
	$(CC) $(CFLAGS) -c main.cpp -o main.o

clean:
	rm -rf ./*.o HW1 ./*.optrpt

run:
	./HW1
