cmake_minimum_required(VERSION 3.14)
project( PolyrepoPracticeCore VERSION 0.0.1 LANGUAGES CXX)

set( CMAKE_CXX_STANDARD 17 )

# Include hipSYCL
find_package(hipSYCL 0.9.1 QUIET)
if (NOT hipSYCL_FOUND)
  message(WARNING "hipsycl not found")
  find_package(IntelDPCPP REQUIRED)
else ()
  message(STATUS "hipsycl found")
  include_directories( ${hipSYCL_INCLUDE_DIRS} )
  set( HIPSYCL_TARGETS "omp" )
endif (NOT hipSYCL_FOUND)

if (IntelDPCPP_FOUND)
  message(STATUS "IntelDPCPP found")
endif (IntelDPCPP_FOUND)

# Set location of custom Boost install:
FIND_PACKAGE( Boost 1.78.0 COMPONENTS REQUIRED )
#INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

# Find Nektar++
FIND_PACKAGE(Nektar++ REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
LINK_DIRECTORIES(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})
MESSAGE(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")

# Set RPATH for wherever Nektar++ is installed.
SET(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")

find_package(MKL CONFIG REQUIRED)
set(CUSTOM_MKL_LINK "-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl")
#message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input
#set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O2 -fPIE" )

# Uncomment this code to list ALL cmake variables
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

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

# Create library
set( LIBRARY_NAME polyrpc )
add_library( ${LIBRARY_NAME} ${LIB_SRCS} )

# Various sanitizers, including coverage and address sanitizer
include(cmake/Sanitizers.cmake)
enable_sanitizers(${LIBRARY_NAME})

# Create executable
add_executable( ${PROJECT_NAME} ${MAIN_SRC} )

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
#target_compile_options( ${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)

# Include src folder for header files
target_include_directories( ${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_include_directories(${LIBRARY_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

# Add libraries
target_link_libraries( ${LIBRARY_NAME} PUBLIC ${BOOST_LIBRARIES} ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES} )
target_link_libraries( ${PROJECT_NAME} PUBLIC ${CUSTOM_MKL_LINK})

# Include test directory
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/test )

# Link main
target_link_libraries( ${PROJECT_NAME} PRIVATE ${LIBRARY_NAME})
target_include_directories( ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )
