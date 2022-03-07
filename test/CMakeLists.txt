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

enable_testing()

file( GLOB TEST_SRCS
	${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp
)

include(GoogleTest)

foreach( TEST ${TEST_SRCS} )
  get_filename_component( EXECUTABLE ${TEST} NAME_WLE )
  message( STATUS "Found test - ${EXECUTABLE}" )
  add_executable( ${EXECUTABLE} ${TEST} )
  include_directories(${Boost_INCLUDE_DIRS})
  target_link_libraries( ${EXECUTABLE} ${LIBRARY_NAME} gtest_main lapack fftw3 )
  target_include_directories( ${EXECUTABLE} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} )
  gtest_discover_tests( ${EXECUTABLE} )
endforeach()

