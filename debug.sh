nvcc -arch=compute_61 -g -o examples/$1 examples/$1.cu src/AI.cu && gdb -ex run ./examples/$1
