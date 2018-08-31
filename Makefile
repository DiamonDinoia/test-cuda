main: main.o
	nvcc -arch=sm_35 main.o -lcudadevrt -o main

main.o: main.cu
	nvcc -arch=sm_35 -dc main.cu -o main.o

main_rdc: main.cu
	nvcc -arch=sm_35 -rdc=true main.cu -lcudadevrt -o main_rdc