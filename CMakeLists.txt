cmake_minimum_required(VERSION 3.30)
project(pstd LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/third_party/stb)
add_executable(${PROJECT_NAME} 
	main.cu
	third_party/stb/stb_image.h
	third_party/stb/stb_image_write.h
	vec3.h
	ray.h
	hitable.h
	sphere.h
	hitable_list.h
	camera.h
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/third_party)