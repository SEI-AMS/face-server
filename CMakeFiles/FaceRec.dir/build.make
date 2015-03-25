# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ben/Desktop/face_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ben/Desktop/face_2

# Include any dependencies generated for this target.
include CMakeFiles/FaceRec.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FaceRec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FaceRec.dir/flags.make

CMakeFiles/FaceRec.dir/face_server.cpp.o: CMakeFiles/FaceRec.dir/flags.make
CMakeFiles/FaceRec.dir/face_server.cpp.o: face_server.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ben/Desktop/face_2/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/FaceRec.dir/face_server.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/FaceRec.dir/face_server.cpp.o -c /home/ben/Desktop/face_2/face_server.cpp

CMakeFiles/FaceRec.dir/face_server.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceRec.dir/face_server.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ben/Desktop/face_2/face_server.cpp > CMakeFiles/FaceRec.dir/face_server.cpp.i

CMakeFiles/FaceRec.dir/face_server.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceRec.dir/face_server.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ben/Desktop/face_2/face_server.cpp -o CMakeFiles/FaceRec.dir/face_server.cpp.s

CMakeFiles/FaceRec.dir/face_server.cpp.o.requires:
.PHONY : CMakeFiles/FaceRec.dir/face_server.cpp.o.requires

CMakeFiles/FaceRec.dir/face_server.cpp.o.provides: CMakeFiles/FaceRec.dir/face_server.cpp.o.requires
	$(MAKE) -f CMakeFiles/FaceRec.dir/build.make CMakeFiles/FaceRec.dir/face_server.cpp.o.provides.build
.PHONY : CMakeFiles/FaceRec.dir/face_server.cpp.o.provides

CMakeFiles/FaceRec.dir/face_server.cpp.o.provides.build: CMakeFiles/FaceRec.dir/face_server.cpp.o

CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o: CMakeFiles/FaceRec.dir/flags.make
CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o: PracticalSocket.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ben/Desktop/face_2/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o -c /home/ben/Desktop/face_2/PracticalSocket.cpp

CMakeFiles/FaceRec.dir/PracticalSocket.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceRec.dir/PracticalSocket.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ben/Desktop/face_2/PracticalSocket.cpp > CMakeFiles/FaceRec.dir/PracticalSocket.cpp.i

CMakeFiles/FaceRec.dir/PracticalSocket.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceRec.dir/PracticalSocket.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ben/Desktop/face_2/PracticalSocket.cpp -o CMakeFiles/FaceRec.dir/PracticalSocket.cpp.s

CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o.requires:
.PHONY : CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o.requires

CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o.provides: CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o.requires
	$(MAKE) -f CMakeFiles/FaceRec.dir/build.make CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o.provides.build
.PHONY : CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o.provides

CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o.provides.build: CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o

# Object files for target FaceRec
FaceRec_OBJECTS = \
"CMakeFiles/FaceRec.dir/face_server.cpp.o" \
"CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o"

# External object files for target FaceRec
FaceRec_EXTERNAL_OBJECTS =

FaceRec: CMakeFiles/FaceRec.dir/face_server.cpp.o
FaceRec: CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o
FaceRec: /usr/local/lib/libopencv_core.a
FaceRec: /usr/local/lib/libopencv_flann.a
FaceRec: /usr/local/lib/libopencv_imgproc.a
FaceRec: /usr/local/lib/libopencv_highgui.a
FaceRec: /usr/local/lib/libopencv_features2d.a
FaceRec: /usr/local/lib/libopencv_calib3d.a
FaceRec: /usr/local/lib/libopencv_cudaarithm.a
FaceRec: /usr/local/lib/libopencv_cudawarping.a
FaceRec: /usr/local/lib/libopencv_ml.a
FaceRec: /usr/local/lib/libopencv_objdetect.a
FaceRec: /usr/local/lib/libopencv_cuda.a
FaceRec: /usr/local/lib/libopencv_cudafilters.a
FaceRec: /usr/local/lib/libopencv_cudaimgproc.a
FaceRec: /usr/local/lib/libopencv_video.a
FaceRec: /usr/local/lib/libopencv_legacy.a
FaceRec: /usr/local/lib/libopencv_cudaoptflow.a
FaceRec: /usr/local/lib/libopencv_photo.a
FaceRec: /usr/local/lib/libopencv_videostab.a
FaceRec: /usr/local/lib/libopencv_ts.a
FaceRec: /usr/local/lib/libopencv_cudacodec.a
FaceRec: /usr/local/lib/libopencv_ocl.a
FaceRec: /usr/local/lib/libopencv_superres.a
FaceRec: /usr/local/lib/libopencv_cudafeatures2d.a
FaceRec: /usr/local/lib/libopencv_nonfree.a
FaceRec: /usr/local/lib/libopencv_stitching.a
FaceRec: /usr/local/lib/libopencv_softcascade.a
FaceRec: /usr/local/lib/libopencv_shape.a
FaceRec: /usr/local/lib/libopencv_optim.a
FaceRec: /usr/local/lib/libopencv_cudastereo.a
FaceRec: /usr/local/lib/libopencv_cudabgsegm.a
FaceRec: /usr/local/lib/libopencv_contrib.a
FaceRec: /usr/local/lib/libopencv_bioinspired.a
FaceRec: /usr/local/lib/libopencv_photo.a
FaceRec: /usr/local/lib/libopencv_cudaoptflow.a
FaceRec: /usr/local/lib/libopencv_cudacodec.a
FaceRec: /usr/local/lib/libopencv_cuda.a
FaceRec: /usr/local/lib/libopencv_cudafeatures2d.a
FaceRec: /usr/local/lib/libopencv_cudawarping.a
FaceRec: /usr/local/lib/libopencv_cudaimgproc.a
FaceRec: /usr/local/lib/libopencv_cudafilters.a
FaceRec: /usr/local/lib/libopencv_legacy.a
FaceRec: /usr/local/lib/libopencv_nonfree.a
FaceRec: /usr/local/lib/libopencv_cudaarithm.a
FaceRec: /usr/local/lib/libopencv_ocl.a
FaceRec: /usr/local/lib/libopencv_calib3d.a
FaceRec: /usr/local/lib/libopencv_features2d.a
FaceRec: /usr/local/lib/libopencv_flann.a
FaceRec: /usr/local/lib/libopencv_objdetect.a
FaceRec: /usr/local/lib/libopencv_highgui.a
FaceRec: /usr/local/share/OpenCV/3rdparty/lib/liblibwebp.a
FaceRec: /usr/lib/x86_64-linux-gnu/libjpeg.so
FaceRec: /usr/lib/x86_64-linux-gnu/libpng.so
FaceRec: /usr/lib/x86_64-linux-gnu/libtiff.so
FaceRec: /usr/lib/x86_64-linux-gnu/libjasper.so
FaceRec: /usr/lib/x86_64-linux-gnu/libjpeg.so
FaceRec: /usr/lib/x86_64-linux-gnu/libpng.so
FaceRec: /usr/lib/x86_64-linux-gnu/libtiff.so
FaceRec: /usr/lib/x86_64-linux-gnu/libjasper.so
FaceRec: /usr/lib/libImath.so
FaceRec: /usr/lib/libIlmImf.so
FaceRec: /usr/lib/libIex.so
FaceRec: /usr/lib/libHalf.so
FaceRec: /usr/lib/libIlmThread.so
FaceRec: /usr/local/lib/libopencv_ml.a
FaceRec: /usr/local/lib/libopencv_video.a
FaceRec: /usr/local/lib/libopencv_imgproc.a
FaceRec: /usr/local/lib/libopencv_core.a
FaceRec: /usr/lib/x86_64-linux-gnu/libz.so
FaceRec: CMakeFiles/FaceRec.dir/build.make
FaceRec: CMakeFiles/FaceRec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable FaceRec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FaceRec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FaceRec.dir/build: FaceRec
.PHONY : CMakeFiles/FaceRec.dir/build

CMakeFiles/FaceRec.dir/requires: CMakeFiles/FaceRec.dir/face_server.cpp.o.requires
CMakeFiles/FaceRec.dir/requires: CMakeFiles/FaceRec.dir/PracticalSocket.cpp.o.requires
.PHONY : CMakeFiles/FaceRec.dir/requires

CMakeFiles/FaceRec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FaceRec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FaceRec.dir/clean

CMakeFiles/FaceRec.dir/depend:
	cd /home/ben/Desktop/face_2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ben/Desktop/face_2 /home/ben/Desktop/face_2 /home/ben/Desktop/face_2 /home/ben/Desktop/face_2 /home/ben/Desktop/face_2/CMakeFiles/FaceRec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FaceRec.dir/depend

