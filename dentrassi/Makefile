UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
	
	LDFLAGS += -lglut -lGL -lGLU -lGLEW -lrt -lm -lasound -lportaudio
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -Xptxas -O3, -g  -Xcompiler "-Wall -fPIC -O3" -ccbin=icc -use_fast_math

all: main.exe 

main.exe: main.o kernel.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h interactions.h
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
