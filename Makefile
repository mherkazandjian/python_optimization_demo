CC = gcc
CFLAGS = -std=c99 -O3 -fopenmp

all: main

laplace_opt.o: laplace_opt.c
	$(CC) $(CFLAGS) -c laplace_opt.c

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

main: main.o laplace_opt.o
	$(CC) $(CFLAGS) -o main main.o laplace_opt.o

run: main
	./main mat.bin 1000 1000

.PHONY: nothing

nothing:

clean:
	rm -f *.o main
	rm -fvr *.so build *~
