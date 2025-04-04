set(EXEC_TARGET_NAME MeshPlotting)

set(MeshPlottingSource ${CMAKE_CURRENT_LIST_DIR}/MeshPlotting.cpp
                       ${CMAKE_CURRENT_LIST_DIR}/MeshPlottingInner.cpp)

add_executable(${EXEC_TARGET_NAME} ${MeshPlottingSource})

# Depending on NESO_LIBRARY_NAME also pulls in all the dependencies like MPI,
# NESO-Particles, Nektar etc.
target_link_libraries(${EXEC_TARGET_NAME} PRIVATE ${NESO_LIBRARY_NAME})
add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES ${MeshPlottingSource})

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
