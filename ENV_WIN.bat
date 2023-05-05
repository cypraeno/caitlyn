@echo off
set PATH=C:/msys64/mingw64/bin;%PATH% # Replace this path with the relevant PATH to your compilers.
set CC=C:/msys64/mingw64/bin/gcc # Replace this with the path to your gcc compiler.
set CXX=C:/msys64/mingw64/bin/g++ # Replace this with the path to your g++ compiler.
set VCPKG_PATH=C:\Users\loico\vcpkg # Replace this with the path to your vcpkg folder.
set TOOLCHAIN_FILE=%VCPKG_PATH%\scripts\buildsystems\vcpkg.cmake
set BUILD_DIR=build
set TARGET_TRIPLET=x64-mingw-static # Replace this with your triplet for compilation. Other examples include MinGW on its own, Visual Studio X, etc.

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

cd %BUILD_DIR%
cmake -G "MinGW Makefiles" -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN_FILE% -DVCPKG_TARGET_TRIPLET=%TARGET_TRIPLET% ..