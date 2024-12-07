nvcc -arch=compute_61 -g -o examples/build/$1 examples/$1.cu && gdb -ex run ./examples/build/$1
