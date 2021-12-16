cmake_minimum_required(VERSION 3.14)
project( PolyrepoPracticeCore VERSION 0.0.1 )

set(CMAKE_CXX_STANDARD 17)
set( HIPSYCL_TARGETS "omp" )

set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3" )

# Get all sources that are not the main
file( GLOB LIB_SRCS ${CMAKE_SOURCE_DIR}/src/*.cpp )
set( MAIN_SRC ${CMAKE_SOURCE_DIR}/src/main.cpp )
list( REMOVE_ITEM LIB_SRCS ${MAIN_SRC} )

# Include LAPACK
find_package(LAPACK)
if( NOT LAPACK_FOUND )
    message( FATAL_ERROR "Could not find required library 'lapack'" )
endif()

# Include hipSYCL
find_package(hipSYCL REQUIRED)

# Create executable
add_executable( ${PROJECT_NAME} ${MAIN_SRC} ${LIB_SRCS} )

# Include src folder for header files
target_include_directories( ${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src )

# Add LAPack libraries
target_link_libraries( ${PROJECT_NAME} PUBLIC ${LAPACK_LIBRARIES} )

add_sycl_to_target( TARGET ${PROJECT_NAME} SOURCES ${MAIN_SRC} )

# Include test directory
add_subdirectory( ${CMAKE_SOURCE_DIR}/test )