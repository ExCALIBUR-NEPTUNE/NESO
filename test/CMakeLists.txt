cmake_minimum_required(VERSION 3.14)
project(PolyrepoPracticeCore VERSION 0.0.1)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 11)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  mesh_test
  mesh_test.cc
  plasma_test
  plasma_test.cc
  fft_test
  fft_test.cc
  ../src/mesh.cpp
  ../src/plasma.cpp
  ../src/fft.cpp
)
target_link_libraries(
  mesh_test
  gtest_main
  lapack
  fftw3
)

include(GoogleTest)
gtest_discover_tests(mesh_test plasma_test fft_test)

