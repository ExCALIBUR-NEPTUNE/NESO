include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file( GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp )

include(GoogleTest)

foreach( TEST ${TEST_SRCS} )
  get_filename_component( EXECUTABLE ${TEST} NAME_WLE )
  message( STATUS "Found test - ${EXECUTABLE}" )
  add_executable( ${EXECUTABLE} ${TEST} )
  target_link_libraries( ${EXECUTABLE} ${LIBRARY_NAME} gtest_main lapack )
  target_include_directories( ${EXECUTABLE} PRIVATE ${CMAKE_SOURCE_DIR}/include )
  gtest_discover_tests( ${EXECUTABLE} )
endforeach()

