cmake_minimum_required(VERSION 3.15)

project(handwritten_numbers LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

add_subdirectory(../.. build)
add_subdirectory(lib/pnglibcpp)

add_executable(training training.cu)
set_target_properties(training PROPERTIES CUDA_ARCHITECTURES all)
target_link_libraries(training PRIVATE ${PNG_LIBRARY})
target_link_libraries(training PRIVATE pnglibcpp SemkiAI)

add_executable(using using.cu)
set_target_properties(using PROPERTIES CUDA_ARCHITECTURES all)
target_link_libraries(using PRIVATE ${PNG_LIBRARY})
target_link_libraries(using PRIVATE pnglibcpp SemkiAI)
