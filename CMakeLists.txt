cmake_minimum_required(VERSION 3.26) # Required

project(Caitlyn VERSION 0.1.0) # Required
set(CMAKE_BUILD_TYPE Release)
set(VCPKG_FEATURE_FLAGS "manifests")

# Find Packages
find_package(SDL2 CONFIG REQUIRED) # SDL2 line 1
add_executable(renderer main.cc)
target_compile_features(renderer PRIVATE cxx_std_11) # Set the C++ standard to C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2") # Set the optimization level to -O2
target_link_libraries(renderer
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)
