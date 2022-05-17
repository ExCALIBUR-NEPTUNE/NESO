cmake_minimum_required(VERSION 3.21)
project(PolyrepoPracticeCore VERSION 0.0.1)

# GoogleTest requires at least C++11
# DataParallel C++ requires at least C++17
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Set location of custom Boost install:
FIND_PACKAGE( Boost 1.78.0 COMPONENTS REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(IntelDPCPP REQUIRED)
find_package(MKL CONFIG REQUIRED)
INCLUDE_DIRECTORIES( $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> )

#message(STATUS "${MKL_IMPORTED_TARGETS}")
#message(STATUS "CMAKE CXX COMPILER : ${CMAKE_CXX_COMPILER}")

enable_testing()

#add_executable(
#  mesh_test
#  mesh_test.cc
#  species_test
#  species_test.cc
#  plasma_test
#  plasma_test.cc
#  fft_mkl_test
#  fft_mkl_test.cc
#  mms_test
#  mms_test.cc
#  diagnostics_test
#  diagnostics_test.cc
#  ../src/mesh.cpp
#  ../src/species.cpp
#  ../src/plasma.cpp
#  ../src/fft_mkl.cpp
#  ../src/diagnostics.cpp
#  ../src/simulation.cpp
#)
#
#target_compile_options(mesh_test PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
#target_include_directories(mesh_test PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
#
#target_link_libraries(
#  mesh_test
#  gtest_main
#  lapack
#  ${Boost_LIBRARIES}
#  sycl
#  ${CUSTOM_MKL_LINK}
#)
#
#enable_testing()

file( GLOB TEST_SRCS
	${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp
)

#target_compile_options(mesh_test PRIVATE -fsycl)
#target_compile_options(mesh_test PUBLIC -fsycl $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)

include(GoogleTest)
#gtest_discover_tests(mesh_test species_test plasma_test mms_test fft_mkl_test)

set(EXECUTABLE testPolyRepoPracticeCore)
message(STATUS "EXECUTABLE : ${EXECUTABLE}")
foreach( TEST ${TEST_SRCS} )
  get_filename_component( EXECUTABLE ${TEST} NAME_WLE )
  message( STATUS "Found test - ${EXECUTABLE}" )
  add_executable( ${EXECUTABLE} ${TEST} )
  include_directories(${Boost_INCLUDE_DIRS})
  target_link_libraries( ${EXECUTABLE} ${LIBRARY_NAME} gtest_main lapack ${CUSTOM_MKL_LINK})
  target_include_directories( ${EXECUTABLE} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> )
  gtest_discover_tests( ${EXECUTABLE} )
endforeach()

