cmake_minimum_required(VERSION 3.30)
project(pstd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
add_subdirectory(third_party)

add_executable(${PROJECT_NAME} 
	main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(${PROJECT_NAME} glm::glm-header-only)
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} glfw)