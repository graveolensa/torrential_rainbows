UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
	LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
	LDFLAGS += -lglut -lGL -lGLU -lGLEW
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -Xptxas -O3, -g  -Xcompiler "-Wall -fPIC -O3 -I/usr/include/c++/7" -ccbin=icc -use_fast_math -maxrregcount=48
INC = -I/usr/local/cuda/samples/common/inc -I/usr/include/c++/8 -I/usr/include/c++/7/

all: main.exe 

main.exe: main.o kernel.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h interactions.h
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
