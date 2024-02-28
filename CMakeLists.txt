cmake_minimum_required(VERSION 3.26) # Required


project(Caitlyn VERSION 0.1.0) # Required
set(CMAKE_BUILD_TYPE Debug)

# EMBREE CONFIG
set(embree_DIR ../opt/lib/cmake/embree-4.3.0/)
# set(TBB_DIR ../opt/lib/cmake/tbb/) haven't installed TBB, doesn't exist
FIND_PACKAGE(embree 4 REQUIRED)

add_executable(caitlyn vec3.cc scene.cc aabb.cc quad.cc main.cc) # Required

# EMBREE CONFIG
TARGET_LINK_LIBRARIES(caitlyn embree)

target_compile_features(caitlyn PUBLIC cxx_std_11) # Set the C++ standard to C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2") # Set the optimization level to -O2

# Instructions
# Create build files:
# mkdir build (if it does not exist already)
# cmake -B build/ -S (DIRECTORY TO CMakeLists.txt)
# All docker tags should have the build folder to hold all the CMake files, do not build files in the same directory 
# as all the code.
# Within build/, CMake files have set up the "make" command:
# make
