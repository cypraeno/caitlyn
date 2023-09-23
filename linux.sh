#!/bin/bash

mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake ..
make