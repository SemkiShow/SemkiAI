cmake_minimum_required(VERSION 3.15)

project(utils)

add_subdirectory(pnglibcpp/)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

add_executable(png_to_csv png_to_csv.cpp)
target_link_libraries(png_to_csv PRIVATE ${PNG_LIBRARY} pnglibcpp)
