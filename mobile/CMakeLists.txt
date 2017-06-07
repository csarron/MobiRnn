cmake_minimum_required(VERSION 3.4.1)

add_library(mobirnn SHARED src/main/cpp/eigen.cc src/main/cpp/opencl.cc)

find_library(log-lib log )

target_link_libraries(mobirnn ${log-lib} )

target_include_directories(mobirnn PRIVATE src/main/cpp/include build/eigen/ build/opencl/)