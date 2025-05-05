cmake_minimum_required(VERSION 3.15)

project(SemkiAI LANGUAGES CXX CUDA)

enable_language(CUDA)

add_library(SemkiAI)

set_target_properties(SemkiAI PROPERTIES CUDA_ARCHITECTURES "52;61;70")

file(GLOB SemkiAISources src/Perceptron.cu src/Perceptron/*.cu)

target_include_directories(SemkiAI PUBLIC src/include/)
target_sources(SemkiAI PRIVATE ${SemkiAISources})

add_library(DatasetParser)

target_include_directories(DatasetParser PUBLIC utils/DatasetParser/include)
target_sources(DatasetParser PRIVATE utils/DatasetParser/DatasetParser.cpp)
