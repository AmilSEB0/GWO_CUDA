cmake_minimum_required(VERSION 3.30)
project(GWO_C)

set(CMAKE_CXX_STANDARD 20)

add_executable(GWO_C main.cpp
        utils/Agent.h
        utils/Agent.cpp
        utils/Problem.h
        utils/Problem.cpp
        OriginalGWO.h
        OriginalGWO.cpp)
