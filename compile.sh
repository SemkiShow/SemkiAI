cmake -B build -DCMAKE_BUILD_TYPE=Release;
cmake --build build --config Release;
# nvcc -arch=compute_61 -o build/bin/SemkiAI src/main.cu src/AI.cu #src/DatasetParser.cpp
./build/bin/SemkiAI
