cmake_minimum_required(VERSION 3.26) # Required


project(Caitlyn VERSION 0.1.3) # Required
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
file(GLOB_RECURSE SRC_FOLDER "src/**/*.cc")
list(APPEND SOURCES ${SRC_FOLDER})
add_executable(caitlyn ${SOURCES}) # Required

# EMBREE CONFIG
TARGET_LINK_LIBRARIES(caitlyn embree)

# Link with libpng
target_link_libraries(caitlyn ${PNG_LIBRARY})
include_directories(caitlyn ${PNG_INCLUDE_DIR})

# Option to disable the building of the csr-validator executable
option(BUILD_CSR_VALIDATOR_EXECUTABLE "Build the CSR Validator executable" OFF)

# Add Submodules
add_subdirectory(csr-schema)
target_link_libraries(caitlyn csr-schema-lib)
target_include_directories(caitlyn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csr-schema/include)

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

# Target all headers recursively in include/ and external/
file(GLOB_RECURSE HEADER_LIST "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE EXTERNAL_HEADER_LIST "${CMAKE_CURRENT_SOURCE_DIR}/external/*.h")

target_include_directories(caitlyn PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# Get the directories of the header files
set(DIR_LIST "")
foreach(FILE_PATH ${HEADER_LIST} ${EXTERNAL_HEADER_LIST})
  get_filename_component(DIR_PATH ${FILE_PATH} PATH)
  list(APPEND DIR_LIST ${DIR_PATH})
endforeach()
list(REMOVE_DUPLICATES DIR_LIST)

# Include the directories
target_include_directories(caitlyn PRIVATE ${DIR_LIST})

# Instructions
# Create build files:
# mkdir build (if it does not exist already)
# cmake -B build/ -S (DIRECTORY TO CMakeLists.txt)
# All docker tags should have the build folder to hold all the CMake files, do not build files in the same directory 
# as all the code.
# Within build/, CMake files have set up the "make" command:
# make
