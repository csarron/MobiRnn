cmake_minimum_required(VERSION 3.4.1)

add_library(main SHARED src/main/cpp/main.cpp)

find_library(log-lib log )

target_link_libraries(main ${log-lib} )

target_include_directories(main PRIVATE build/eigen/)