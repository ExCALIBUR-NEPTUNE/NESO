# Identify source files
set(DRIFTREDUCED_SRC_FILES
    EquationSystems/DriftReducedSystem.cpp
    EquationSystems/HWSystem.cpp
    EquationSystems/HW2DSystem.cpp
    EquationSystems/HW3DSystem.cpp
    EquationSystems/LAPDSystem.cpp
    EquationSystems/RogersRicci2DSystem.cpp
    ParticleSystems/NeutralParticleSystem.cpp)

# ============================== Object library ===============================
# Put solver specific source in an object library so that tests can use it
set(LIBRARY_NAME DriftReduced_ObjLib)
set(SOLVER_LIBS
    ${SOLVER_LIBS} ${LIBRARY_NAME}
    CACHE INTERNAL "")
add_library(${LIBRARY_NAME} OBJECT ${DRIFTREDUCED_SRC_FILES})
target_compile_options(${LIBRARY_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
target_link_libraries(
  ${LIBRARY_NAME} PRIVATE Nektar++::nektar++ NESO-Particles::NESO-Particles
                          ${NESO_LIBRARY_NAME})

# Allow this solver's headers to be included from the tests
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_sycl_to_target(TARGET ${LIBRARY_NAME} SOURCES ${DRIFTREDUCED_SRC_FILES})

# =================================== Exec ====================================
set(EXEC_TARGET_NAME DriftReduced)

add_executable(${EXEC_TARGET_NAME} ../main.cpp
                                   $<TARGET_OBJECTS:${LIBRARY_NAME}>)

# Compile options
target_compile_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})

# Linker options, target libs
target_link_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_LINK_FLAGS})
target_link_libraries(
  ${EXEC_TARGET_NAME} PRIVATE Nektar++::nektar++ NESO-Particles::NESO-Particles
                              ${NESO_LIBRARY_NAME})

add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES main.cpp)

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Exec rpath
set_target_properties(
  ${EXEC_TARGET_NAME} PROPERTIES INSTALL_RPATH ${INSTALL_RPATH}
                                 ${NEKTAR++_LIBRARY_DIRS})
