include(FetchContent)
#Get googlebenchmark suite
option(BENCHMARK_DOWNLOAD_DEPENENCIES ON)
option(BENCHMARK_ENABLE_GTEST_TESTS OFF)
option(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(benchmark
    GIT_SHALLOW TRUE
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG main #TODO: Change this to a specific commit
    )
FetchContent_MakeAvailable(benchmark)

set(PROJECTION_SRC projection/args.cpp projection/benchmark.cpp)

add_executable(projection_bench 
	${PROJECTION_SRC})
add_sycl_to_target(TARGET projection_bench SOURCE ${PROJECTION_BENCH})
target_link_libraries(projection_bench 
PUBLIC
	benchmark::benchmark
	${NESO_LIBRARY_NAME}
)


