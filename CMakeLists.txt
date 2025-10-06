cmake_minimum_required(VERSION 3.30)
project(pstd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(${PROJECT_NAME} 
	main.cpp
)
