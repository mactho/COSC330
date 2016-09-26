COMPILER = nvcc
CFLAGS = -I /usr/local/cuda-7.5/samples/common/inc
COBJS = utils.o
CEXES =  start

all: ${CEXES}

start: start.cu ${COBJS}
	${COMPILER} ${CFLAGS} start.cu ${COBJS} -o start -lm

%.o: %.cu %.h  makefile
	${COMPILER} ${CFLAGS} -lm $< -c

clean:
	rm -f *.o *~ ${CEXES} output.csv
