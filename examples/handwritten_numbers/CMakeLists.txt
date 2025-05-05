cmake_minimum_required(VERSION 3.15)

project(handwritten_numbers LANGUAGES CXX CUDA)

enable_language(CUDA)

add_subdirectory(../.. build)

add_executable(training training.cu)
set_target_properties(training PROPERTIES CUDA_ARCHITECTURES "52;61;70")
target_link_libraries(training PRIVATE SemkiAI DatasetParser)

add_executable(using using.cu)
set_target_properties(using PROPERTIES CUDA_ARCHITECTURES "52;61;70")
target_link_libraries(using PRIVATE SemkiAI DatasetParser)
