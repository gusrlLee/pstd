cmake_minimum_required(VERSION 3.30)
project(pstd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_subdirectory(ext)

add_executable(${PROJECT_NAME} 
	src/pstd.hpp
	src/main.cpp
	src/types.hpp

	src/context.hpp
	src/context.cpp

	src/utils.hpp
	src/utils.cpp

	src/structs.hpp
)

target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} glfw)
target_link_libraries(${PROJECT_NAME} glm::glm-header-only)
target_link_libraries(${PROJECT_NAME} GPUOpen::VulkanMemoryAllocator)
target_link_libraries(${PROJECT_NAME} fmt::fmt-header-only)