# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/supremacy/Desktop/code/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/supremacy/Desktop/code/project/build

# Include any dependencies generated for this target.
include CMakeFiles/fps.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fps.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fps.dir/flags.make

CMakeFiles/fps.dir/src/Fps.cpp.o: CMakeFiles/fps.dir/flags.make
CMakeFiles/fps.dir/src/Fps.cpp.o: ../src/Fps.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/supremacy/Desktop/code/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fps.dir/src/Fps.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fps.dir/src/Fps.cpp.o -c /home/supremacy/Desktop/code/project/src/Fps.cpp

CMakeFiles/fps.dir/src/Fps.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fps.dir/src/Fps.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/supremacy/Desktop/code/project/src/Fps.cpp > CMakeFiles/fps.dir/src/Fps.cpp.i

CMakeFiles/fps.dir/src/Fps.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fps.dir/src/Fps.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/supremacy/Desktop/code/project/src/Fps.cpp -o CMakeFiles/fps.dir/src/Fps.cpp.s

# Object files for target fps
fps_OBJECTS = \
"CMakeFiles/fps.dir/src/Fps.cpp.o"

# External object files for target fps
fps_EXTERNAL_OBJECTS =

../lib/libfps.so: CMakeFiles/fps.dir/src/Fps.cpp.o
../lib/libfps.so: CMakeFiles/fps.dir/build.make
../lib/libfps.so: CMakeFiles/fps.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/supremacy/Desktop/code/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../lib/libfps.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fps.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fps.dir/build: ../lib/libfps.so

.PHONY : CMakeFiles/fps.dir/build

CMakeFiles/fps.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fps.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fps.dir/clean

CMakeFiles/fps.dir/depend:
	cd /home/supremacy/Desktop/code/project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/supremacy/Desktop/code/project /home/supremacy/Desktop/code/project /home/supremacy/Desktop/code/project/build /home/supremacy/Desktop/code/project/build /home/supremacy/Desktop/code/project/build/CMakeFiles/fps.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fps.dir/depend

