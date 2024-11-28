# cmake -B build -DCMAKE_BUILD_TYPE=Release;
# cmake --build build --config Release;
nvcc -arch=compute_61 -o build/bin/SemkiAI src/*.cu && time ./build/bin/SemkiAI #src/DatasetParser.cpp
# hipcc -arch=compute_61 -o build/bin/SemkiAI src/main.cu src/AI.cu #src/DatasetParser.cpp
# hipcc -arch=compute_61 -o build/bin/SemkiAI src/main.hip src/AI.hip #src/DatasetParser.cpp
# time ./build/bin/SemkiAI
