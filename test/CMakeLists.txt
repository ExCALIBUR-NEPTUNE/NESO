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

set(EXECUTABLE testPolyRepoPracticeCore)

add_executable(
  ${EXECUTABLE}
  test_mesh.cpp
  test_species.cpp
  test_plasma.cpp
  test_fft_mkl.cpp
  test_mms.cpp
  test_diagnostics.cpp
  ../src/mesh.cpp
  ../src/species.cpp
  ../src/plasma.cpp
  ../src/fft_mkl.cpp
  ../src/diagnostics.cpp
  ../src/simulation.cpp
)

target_compile_options(${EXECUTABLE} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(${EXECUTABLE} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(
  ${EXECUTABLE}
  gtest_main
  lapack
  ${Boost_LIBRARIES}
  sycl
  ${CUSTOM_MKL_LINK}
)

target_compile_options(${EXECUTABLE} PUBLIC -fsycl $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)

include(GoogleTest)
gtest_discover_tests(${EXECUTABLE} species_test plasma_test fft_test fft_3d_test mms_test)


###enable_testing()
###
###set(EXECUTABLE testPolyRepoPracticeCore)
###
###file( GLOB TEST_SRCS
###	${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
###	${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp
###)
###
###include(GoogleTest)
###message(STATUS "EXECUTABLE : ${EXECUTABLE}")
###foreach( TEST ${TEST_SRCS} )
###  get_filename_component( EXECUTABLE ${TEST} NAME_WLE )
###  message( STATUS "Found test - ${EXECUTABLE}" )
###  add_executable( ${EXECUTABLE} ${TEST} )
###  include_directories(${Boost_INCLUDE_DIRS})
###  target_link_libraries( ${EXECUTABLE} ${LIBRARY_NAME} gtest_main lapack ${CUSTOM_MKL_LINK})
###  target_include_directories( ${EXECUTABLE} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> )
###  target_compile_options(${EXECUTABLE} PUBLIC -fsycl $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
###  gtest_discover_tests( ${EXECUTABLE} )
###endforeach()
###
