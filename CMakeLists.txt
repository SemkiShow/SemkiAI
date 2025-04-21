cmake_minimum_required(VERSION 3.15)

project(SemkiAI LANGUAGES CXX CUDA)

enable_language(CUDA)

add_library(SemkiAI)

set_target_properties(SemkiAI PROPERTIES CUDA_ARCHITECTURES "52;70")

target_include_directories(SemkiAI PUBLIC src/include/)
target_sources(SemkiAI PRIVATE src/Perceptron.cu src/DatasetParser.cpp)

add_library(DatasetParser)

target_include_directories(DatasetParser PUBLIC src/include)
target_sources(DatasetParser PRIVATE src/DatasetParser.cpp)
