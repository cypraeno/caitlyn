#!/bin/bash

# Run this from inside the tools folder, as the following paths are RELATIVE to your
# current location

# Defaults for if caitlyn is in the build folder. Create a test_outputs folder in build/.
test_outputs_dir="../build/test_outputs"
tests_dir="../tests"
executable="../build/caitlyn"

if [ "$1" == 'yes' ]; then
    # Navigate to the build directory
    pushd ../build > /dev/null

    make clean
    make
    
    # Return to the original directory
    popd > /dev/null
fi

for file in "$tests_dir"/*.csr; do
    echo "$file"
    "$executable" -i "$file" -t png -o "$test_outputs_dir/$(basename "$file" .csr).png" -s 2 -d 2 -V
done