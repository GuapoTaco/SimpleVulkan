cmake_minimum_required(VERSION 2.6)
project(vulkan101)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS -fsanitize=address)

find_library(OpenGL REQUIRED)
find_library(glfw REQUIRED)
find_program(GLSLANG glslangValidator)

set(SRCS 
	main.cpp
)

add_custom_target(shaders ALL
	COMMAND ${GLSLANG} -V ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert
	DEPENDS shader.frag
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Compiling GLSL shaders"
	SOURCES shader.frag shader.vert
	
)

add_executable(vulkan101 ${SRCS})

target_link_libraries(vulkan101 ${OpenGL_LIBRARIES} glfw3 X11 Xrandr Xxf86vm Xcursor Xinerama vulkan)

target_compile_features(vulkan101 PUBLIC cxx_variable_templates)
