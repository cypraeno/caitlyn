#!/bin/bash

# Defaults for if caitlyn is in the build folder. Create a test_outputs folder in build/.
test_outputs_dir="../build/test_outputs"
tests_dir="./"
executable="../build/caitlyn"

if [ $1 == 'yes' ]
then
    make clean
    make
fi

for file in "$tests_dir"/*.csr; do
    "$executable" -i "$file" -t png -o "$test_outputs_dir/$(basename "$file" .csr).png" -s 2 -d 2 -V
done