nvcc -arch=compute_61 -g -o build/bin/SemkiAI src/*.cu && gdb -ex run ./build/bin/SemkiAI
