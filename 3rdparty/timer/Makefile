O  = o
A  = a
cc = /usr/bin/gcc 
CC = /usr/bin/g++
AR = ar ruvs
LK = ${CC} ${LF} -o
SH = ${CC} -dynamiclib -all_load ${LF} -o


CUR_DIR = $(shell pwd)

GEN_LIB    = ${CUR_DIR}/libtimer.$A
GEN_SO_LIB = ${CUR_DIR}/libtimer.so

SRC_FILES = $(shell find . -type f -name '*.cpp')
OBJ_FILES = $(SRC_FILES:.cpp=.$O)

CF += -O3 -DNDEBUG
LF +=

LIBS = ${OPEN_CV_LIBS}

all: ${OBJ_FILES}
	$(AR) ${GEN_LIB} ${OBJ_FILES}
	${SH} ${GEN_SO_LIB} ${GEN_LIB} ${LIBS}

clean:
	rm -f $(OBJ_FILES)
	rm -f ${GEN_LIB} ${GEN_SO_LIB}


# Rules to compile files
.SUFFIXES: .$O .o .c .C .cpp .cc

.c.$O:
	${CC} -x c++ ${CF} -c -o  $*.$O $*.c
	
.cc.$O:
	${CC} -x c++ ${CF} -c -o  $*.$O $*.cc

.C.$O:
	${CC} -x c++ ${CF} -c -o  $*.$O $*.C

.cpp.$O:
	${CC} -x c++ ${CF} -c -o  $*.$O $*.cpp
