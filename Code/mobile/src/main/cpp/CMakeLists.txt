cmake_minimum_required(VERSION 3.4.1)

add_library(main SHARED main.cpp math.cpp)

find_library(log-lib log )

target_link_libraries(main ${log-lib} )