cmake_minimum_required(VERSION 3.30)
project(pstd LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(${PROJECT_NAME} main.cu)