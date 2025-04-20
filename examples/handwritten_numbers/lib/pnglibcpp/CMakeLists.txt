cmake_minimum_required(VERSION 3.15)

project(pnglibcpp)

add_library(pnglibcpp)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

target_include_directories(pnglibcpp PUBLIC .)
target_sources(pnglibcpp PRIVATE pnglib.cpp)
target_link_libraries(pnglibcpp PRIVATE ${PNG_LIBRARY})
