# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /app

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /app/build

# Include any dependencies generated for this target.
include CMakeFiles/caitlyn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/caitlyn.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/caitlyn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/caitlyn.dir/flags.make

CMakeFiles/caitlyn.dir/main.cc.o: CMakeFiles/caitlyn.dir/flags.make
CMakeFiles/caitlyn.dir/main.cc.o: /app/main.cc
CMakeFiles/caitlyn.dir/main.cc.o: CMakeFiles/caitlyn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/caitlyn.dir/main.cc.o"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/caitlyn.dir/main.cc.o -MF CMakeFiles/caitlyn.dir/main.cc.o.d -o CMakeFiles/caitlyn.dir/main.cc.o -c /app/main.cc

CMakeFiles/caitlyn.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caitlyn.dir/main.cc.i"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /app/main.cc > CMakeFiles/caitlyn.dir/main.cc.i

CMakeFiles/caitlyn.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caitlyn.dir/main.cc.s"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /app/main.cc -o CMakeFiles/caitlyn.dir/main.cc.s

# Object files for target caitlyn
caitlyn_OBJECTS = \
"CMakeFiles/caitlyn.dir/main.cc.o"

# External object files for target caitlyn
caitlyn_EXTERNAL_OBJECTS =

caitlyn: CMakeFiles/caitlyn.dir/main.cc.o
caitlyn: CMakeFiles/caitlyn.dir/build.make
caitlyn: /opt/lib/libembree4.so.4
caitlyn: CMakeFiles/caitlyn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable caitlyn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caitlyn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/caitlyn.dir/build: caitlyn
.PHONY : CMakeFiles/caitlyn.dir/build

CMakeFiles/caitlyn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/caitlyn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/caitlyn.dir/clean

CMakeFiles/caitlyn.dir/depend:
	cd /app/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /app /app /app/build /app/build /app/build/CMakeFiles/caitlyn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/caitlyn.dir/depend

