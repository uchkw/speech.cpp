# The build of the 032b experiment, a small diagnostic project beside the complete repository: it adds the
# repository as a subdirectory and links the family and common libraries that its CMakeLists defines, changing none
# of its targets and none of its files. The build directory is the experiment's own, so a main build beside it is
# not overwritten, and the configure and the build take the campaign's lease, which keeps one such run on the
# machine at a time; from this folder:
#
#     /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake -B build-032b -S .
#     /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake --build build-032b -j4 --target fastconformer-batch2-check fastconformer-batch2-model
#
# Nothing here is configured or built by the repository's own build, and this file is the experiment's alone.

cmake_minimum_required(VERSION 3.20)
project(fastconformer-batch2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The sources are UTF-8; MSVC otherwise reads them in the system code page.
if (MSVC)
    add_compile_options(/utf-8)
endif()

# The complete speech.cpp repository, three folders up, its targets excluded from this project's "all" so that only
# the two checks and what they link are built.
get_filename_component(SPEECH_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
add_subdirectory(${SPEECH_CPP_SOURCE} speech-cpp EXCLUDE_FROM_ALL)

# The synthetic differential check, which writes its own model file into a work dir.
add_executable(fastconformer-batch2-check batch-check.cpp batch-encoder.cpp fixture.cpp)
target_include_directories(fastconformer-batch2-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${SPEECH_CPP_SOURCE}/checks)
target_link_libraries(fastconformer-batch2-check PRIVATE fastconformer)

# The real-model check and timing entry point, which takes a model file and two recordings.
add_executable(fastconformer-batch2-model batch-model.cpp batch-encoder.cpp)
target_include_directories(fastconformer-batch2-model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${SPEECH_CPP_SOURCE}/checks)
target_link_libraries(fastconformer-batch2-model PRIVATE fastconformer)
