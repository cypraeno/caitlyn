cmake_minimum_required(VERSION 3.26) # Required


project(Caitlyn VERSION 0.1.0) # Required
set(CMAKE_BUILD_TYPE Debug)

# Include SIMD Configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/SIMDConfig.cmake)

# EMBREE CONFIG
set(embree_DIR ../opt/lib/cmake/embree-4.3.0/)
# set(TBB_DIR ../opt/lib/cmake/tbb/) haven't installed TBB, doesn't exist
FIND_PACKAGE(embree 4 REQUIRED)

# Find libpng package
find_package(PNG REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")
add_executable(caitlyn ${SOURCES}) # Required

# EMBREE CONFIG
TARGET_LINK_LIBRARIES(caitlyn embree)

# Link with libpng
target_link_libraries(caitlyn ${PNG_LIBRARY})
include_directories(${PNG_INCLUDE_DIR})

target_compile_features(caitlyn PUBLIC cxx_std_11) # Set the C++ standard to C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2") # Set the optimization level to -O2

# Set EMBREE_MAX_ISA based on compiler support
if(COMPILER_SUPPORTS_AVX2)
  target_compile_definitions(caitlyn PRIVATE EMBREE_MAX_ISA=AVX2)
elseif(COMPILER_SUPPORTS_AVX)
  target_compile_definitions(caitlyn PRIVATE EMBREE_MAX_ISA=AVX)
elseif(COMPILER_SUPPORTS_SSE42)
  target_compile_definitions(caitlyn PRIVATE EMBREE_MAX_ISA=SSE4.2)
endif()

target_include_directories(caitlyn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# THIS SHOULD BE REMOVED AFTER ALL NECESSARY INCLUDES ARE MOVED INTO include/ IN THE FUTURE
target_include_directories(caitlyn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Instructions
# Create build files:
# mkdir build (if it does not exist already)
# cmake -B build/ -S (DIRECTORY TO CMakeLists.txt)
# All docker tags should have the build folder to hold all the CMake files, do not build files in the same directory 
# as all the code.
# Within build/, CMake files have set up the "make" command:
# make
