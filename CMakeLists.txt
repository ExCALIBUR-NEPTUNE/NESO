cmake_minimum_required(VERSION 3.14)

project(
  NESO
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Include hipSYCL
find_package(hipSYCL 0.9.1 QUIET)
if(NOT hipSYCL_FOUND)
  message(WARNING "hipsycl not found")
  find_package(IntelDPCPP REQUIRED)
else()
  message(STATUS "hipsycl found")
  include_directories(${hipSYCL_INCLUDE_DIRS})
  set(HIPSYCL_TARGETS "omp")
  ADD_DEFINITIONS(-DNESO_HIPSYCL)
endif(NOT hipSYCL_FOUND)

if(IntelDPCPP_FOUND)
  message(STATUS "IntelDPCPP found")
  ADD_DEFINITIONS(-DNESO_DPCPP)
endif(IntelDPCPP_FOUND)

# hipsycl, trisycl and computecpp all define an "add_sycl_to_target" for the
# compilation of a target
if(NOT COMMAND add_sycl_to_target)
    # Note from hipsycl: "Do not call target_sources after add_sycl_to_target
    # or dependency tracking on compiler flags will break in subtle ways"
    function(add_sycl_to_target)
    endfunction()
endif()

# Set location of custom Boost install:
find_package(
  Boost 1.78.0
  COMPONENTS
  REQUIRED)
# INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

# Find Nektar++
find_package(Nektar++ REQUIRED)
include_directories(SYSTEM ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})
message(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")

# Set RPATH for wherever Nektar++ is installed.
set(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")


find_package(MKL CONFIG QUIET)
if (MKL_FOUND)
    # message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets
    # based on input set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O2 -fPIE" )
    set(CUSTOM_MKL_LINK "-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl")
    ADD_DEFINITIONS(-DNESO_INTEL_MKL_FFT)
else()
    find_package(PkgConfig REQUIRED)
    pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
    ADD_DEFINITIONS(-DNESO_FFTW_FFT)
endif()


# Uncomment this code to list ALL cmake variables
# get_cmake_property(_variableNames VARIABLES) list (SORT _variableNames)
# foreach (_variableName ${_variableNames}) message(STATUS
# "${_variableName}=${${_variableName}}") endforeach()

enable_testing()

# put all targets in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# put all libraries in lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ##############################################################################
# Set the build type environment variable
# ##############################################################################
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
file(GLOB LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
list(REMOVE_ITEM LIB_SRCS ${MAIN_SRC})

# Create library
set(LIBRARY_NAME polyrpc)
add_library(${LIBRARY_NAME} ${LIB_SRCS})

include(src/nektar/CMakeLists.txt)



# Various sanitizers, including coverage and address sanitizer
include(cmake/Sanitizers.cmake)
enable_sanitizers(${LIBRARY_NAME})

# Get the git commit
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC NESO_REVISION)
if(NESO_REVISION STREQUAL "GITDIR-NOTFOUND")
  set(NESO_REVISION "Unknown")
endif()
message(STATUS "Git revision: ${NESO_REVISION}")
git_local_changes(NESO_GIT_STATE)
message(STATUS "Git repo state: ${NESO_GIT_STATE}")

# Build the file containing just the commit hash This will be rebuilt on every
# commit!
configure_file("${PROJECT_SOURCE_DIR}/include/revision.hpp.in"
               "${PROJECT_SOURCE_DIR}/include/revision.hpp")
# Create executable
add_executable(${PROJECT_NAME} ${MAIN_SRC})

# define the source file objects as a sycl targets
add_sycl_to_target(TARGET ${LIBRARY_NAME} SOURCES ${LIB_SRCS})

message("TODO UNCOMMMENT WERROR")
message("TODO UNCOMMMENT WERROR")
message("TODO UNCOMMMENT WERROR")
message("TODO UNCOMMMENT WERROR")
message("TODO UNCOMMMENT WERROR")
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
#target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
# target_compile_options( ${PROJECT_NAME} PUBLIC
# $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)

# Include src folder for header files
target_include_directories(${LIBRARY_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (MKL_FOUND)
    target_include_directories(${LIBRARY_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
if (FFTW_FOUND)
    include_directories(${LIBRARY_NAME} PUBLIC PkgConfig::FFTW)
    include_directories(${PROJECT_NAME} PUBLIC PkgConfig::FFTW)
endif()



# Add libraries
target_link_libraries( ${LIBRARY_NAME} PUBLIC ${BOOST_LIBRARIES} ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES} )
if (MKL_FOUND)
    target_link_libraries( ${PROJECT_NAME} PUBLIC ${CUSTOM_MKL_LINK})
endif()
if (FFTW_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC PkgConfig::FFTW)
endif()

# Include test directory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)

# Link main
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_NAME})
target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# define the main executable as a sycl target
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${MAIN_SRC})
