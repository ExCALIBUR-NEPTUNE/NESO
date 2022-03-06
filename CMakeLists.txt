cmake_minimum_required(VERSION 3.14)
project( PolyrepoPracticeCore VERSION 0.0.1 )

set( CMAKE_CXX_STANDARD 17 )

# Include hipSYCL
find_package(hipSYCL 0.9.1 QUIET)
if (NOT hipSYCL_FOUND)
  find_package(IntelDPCPP REQUIRED)
else ()
  set( HIPSYCL_TARGETS "omp" )
endif (NOT hipSYCL_FOUND)

#set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O2 -fPIE" )

enable_testing()

## put all targets in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
## put all libraries in lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

############################################################
# Set the build type environment variable
############################################################
if(BUILD_TYPE STREQUAL "TEST")
  set(BUILD_TYPE_COMPILE_FLAGS "-g;-O0;--coverage")
  set(TEST_LIBRARIES "gcov")
elseif(BUILD_TYPE STREQUAL "DEBUG")
  set(BUILD_TYPE_COMPILE_FLAGS "-g -O0")
  set(TEST_LIBRARIES "")
elseif(BUILD_TYPE STREQUAL "RELEASE")
  set(BUILD_TYPE_COMPILE_FLAGS "-O2")
  set(TEST_LIBRARIES "")
endif()

# Get all sources that are not the main
file( GLOB LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp )
set( MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp )
list( REMOVE_ITEM LIB_SRCS ${MAIN_SRC} )

# Include LAPACK
find_package(LAPACK)
if( NOT LAPACK_FOUND )
    message( FATAL_ERROR "Could not find required library 'lapack'" )
endif()

# Include Boost
# Set location of custom Boost install:
set(Boost_INCLUDE_DIR /root/code/boost_1_78_0)
FIND_PACKAGE( Boost 1.78.0 COMPONENTS REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

# Include FFTW - the standard way of finding fftw is broken upstream
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)
link_libraries(PkgConfig::FFTW)

# Create library
set( LIBRARY_NAME polyrpc )
add_library( ${LIBRARY_NAME} ${LIB_SRCS} )

# Create executable
add_executable( ${PROJECT_NAME} ${MAIN_SRC} )

# Include src folder for header files
target_include_directories( ${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )

# Add libraries
target_link_libraries( ${LIBRARY_NAME} PUBLIC ${LAPACK_LIBRARIES} ${BOOST_LIBRARIES} )

# Include test directory
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/test )

# Link main
#add_sycl_to_target( TARGET ${PROJECT_NAME} SOURCES ${MAIN_SRC} )
target_link_libraries( ${PROJECT_NAME} PRIVATE ${LIBRARY_NAME} )
target_include_directories( ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )
