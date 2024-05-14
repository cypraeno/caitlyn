#!/bin/bash
cd ..
mkdir build
cmake -B build/ -S .
cd build
make