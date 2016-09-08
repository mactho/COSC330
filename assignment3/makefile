COMPILER = gcc
CFLAGS = -Wall -pedantic
COBJS = gaussianLib.o qdbmp.o
CEXES =  gaussian

all: ${CEXES}

gaussian: gaussian.c ${COBJS}
	${COMPILER} ${CFLAGS} gaussian.c ${COBJS} -o gaussian -lm

%.o: %.c %.h  makefile
	${COMPILER} ${CFLAGS} -lm $< -c

clean:
	rm -f *.o *~ ${CEXES}
