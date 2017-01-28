CC= gcc
CFLAGS= -O3 -msse4.1

all: mm

mm: mm.o
	$(CC) $(CFLAGS) -o mm mm.o 

mm.o: mm.c
	$(CC) $(CFLAGS) -I. -c mm.c

clean:
	rm -f *.o mm
