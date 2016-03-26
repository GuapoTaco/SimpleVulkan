cmake_minimum_required(VERSION 2.6)
project(vulkan101)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS -fsanitize=address)

find_package(Threads REQUIRED)
find_package(vulkan REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

find_program(GLSLANG glslangValidator)

include_directories(${VULKAN_INCLUDE_DIR})

set(SRCS 
	main.cpp
	lodepng.cpp
)

add_custom_target(shaders ALL
	COMMAND ${GLSLANG} -V ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert
	DEPENDS shader.frag shader.vert
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Compiling GLSL shaders"
	SOURCES shader.frag shader.vert
	
)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/image.png" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}) 

add_executable(vulkan101 ${SRCS})

target_link_libraries(vulkan101 ${VULKAN_LIBRARY} ${GLFW_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if(CMAKE_SYSTEM_NAME STREQUAL  Linux)
	target_link_libraries(vulkan101 X11 Xrandr dl Xxf86vm Xinerama Xcursor)
endif()

target_compile_features(vulkan101 PUBLIC cxx_variable_templates)
