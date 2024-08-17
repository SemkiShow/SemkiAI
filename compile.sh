# cmake -B build -DCMAKE_BUILD_TYPE=Release;
# cmake --build build --config Release;
hipcc -arch=compute_61 -o build/bin/SemkiAI src/main.cu src/AI.cu #src/DatasetParser.cpp
# time ./build/bin/SemkiAI
