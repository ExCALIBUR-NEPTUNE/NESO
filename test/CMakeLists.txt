cmake_minimum_required(VERSION 3.21)
project(NESO VERSION 0.0.1)

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

find_package(MKL CONFIG REQUIRED)
INCLUDE_DIRECTORIES( $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> )

enable_testing()

set(EXECUTABLE testNESO)

# List test source files
file( GLOB TEST_SRCS
	${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp
)
message(STATUS ${TEST_SRCS})

# Build the tests individually
include(GoogleTest)
foreach( TEST ${TEST_SRCS} )
	get_filename_component( TEST_NAME ${TEST} NAME_WLE )
  	message( STATUS "Found test - ${TEST_NAME}" )
	set(TEST_LIST ${TEST_LIST} ${TEST_NAME})
	#message( STATUS "Test list - ${TEST_LIST}" )
	add_executable( ${TEST_NAME} ${TEST} )
	include_directories(${Boost_INCLUDE_DIRS})
	target_link_libraries( ${TEST_NAME} ${LIBRARY_NAME} gtest_main lapack ${CUSTOM_MKL_LINK})
	target_include_directories( ${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> )
	target_compile_options(${TEST_NAME} PUBLIC -fsycl $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
	gtest_discover_tests( ${TEST_NAME} )
	
endforeach()

# Build a global test suite
add_executable(
  ${EXECUTABLE}
  ${TEST_LIST}
  ../src/mesh.cpp
  ../src/species.cpp
  ../src/plasma.cpp
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


gtest_discover_tests(${EXECUTABLE})
