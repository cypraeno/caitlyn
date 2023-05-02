# This script is meant for developers to easily set up the environment and build files. 

#!/bin/sh
# chmod +x configure.sh

VCPKG_PATH="/Users/MonkeyDumpling/vcpkg" # Replace this with the actual path to the vcpkg directory
TOOLCHAIN_FILE="$VCPKG_PATH/scripts/buildsystems/vcpkg.cmake"
BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

cd "$BUILD_DIR"
cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" ..