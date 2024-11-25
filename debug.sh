nvcc -arch=compute_61 -g -o build/bin/SemkiAI src/main.cu src/AI.cu && gdb -ex run ./build/bin/SemkiAI
