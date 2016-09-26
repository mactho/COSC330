COMPILER = gcc
CFLAGS = -Wall -pthread -pedantic
CEXES = nordvic

all: ${CEXES}

nordvic: nordvic.c
	${COMPILER} ${CFLAGS} nordvic.c -o nordvic

clean:
	rm -f *.o *~ ${CEXES}
