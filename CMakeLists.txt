cmake_minimum_required(VERSION 3.15)

project(SemkiAI LANGUAGES CXX CUDA)

enable_language(CUDA)

add_library(SemkiAI)

set_target_properties(SemkiAI PROPERTIES CUDA_ARCHITECTURES all)

target_include_directories(SemkiAI PUBLIC src/include/)
target_sources(SemkiAI PRIVATE src/Perceptron.cu)
