UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
	LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
	LDFLAGS += -lglut -lGL -lGLU -lGLEW
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -fPIC -O3 -parallel" -ccbin=icc -use_fast_math -gencode=arch=compute_61,code=sm_61 -O2
INC = -I/usr/local/cuda/samples/common/inc

all: main.exe 

main.exe: main.o kernel.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h interactions.h
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
