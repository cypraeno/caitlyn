@echo off

set VCPKG_PATH=C:\path\to\vcpkg
set TOOLCHAIN_FILE=%VCPKG_PATH%\scripts\buildsystems\vcpkg.cmake
set BUILD_DIR=build

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

cd %BUILD_DIR%
cmake -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN_FILE% ..