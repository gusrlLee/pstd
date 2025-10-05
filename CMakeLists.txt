cmake_minimum_required(VERSION 3.30)
project(pstd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_ARCHITECTURES native)

add_subdirectory(third_party)

include_directories(${CMAKE_SOURCE_DIR}/third_party)

add_executable(${PROJECT_NAME} 
	main.cu

	triangle.h
	vec3.h
	ray.h
	hitable.h
	sphere.h
	hitable_list.h
	camera.h
	material.h
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/third_party)