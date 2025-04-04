# Identify source files
set(DIFFSOLVER_SRC_FILES EquationSystems/DiffusionSystem.cpp)
# Building cwipi equation systems is optional
if(NESO_BUILD_CWIPI_EXAMPLES)
  set(DIFFSOLVER_SRC_FILES
      EquationSystems/CwipiDiffTensorSender.cpp
      EquationSystems/CwipiReceiveDiffTensorAndDiffuse.cpp
      ${DIFFSOLVER_SRC_FILES})
endif()

# ============================== Object library ===============================
# Put solver specific source in an object library so that tests can use it
set(LIBRARY_NAME Diffusion_ObjLib)
set(SOLVER_LIBS
    ${SOLVER_LIBS} ${LIBRARY_NAME}
    CACHE INTERNAL "")
add_library(${LIBRARY_NAME} OBJECT ${DIFFSOLVER_SRC_FILES})

# Compile options and definitions
target_compile_definitions(${LIBRARY_NAME} PUBLIC ${BUILD_TYPE_DEFINITIONS})
target_compile_options(${LIBRARY_NAME} PUBLIC ${BUILD_TYPE_COMPILE_FLAGS})

# Linked libraries
target_link_libraries(
  ${LIBRARY_NAME} PUBLIC Nektar++::nektar++ NESO-Particles::NESO-Particles
                         ${NESO_LIBRARY_NAME})

# Enable sycl
add_sycl_to_target(TARGET ${LIBRARY_NAME} SOURCES ${DIFFSOLVER_SRC_FILES})

# =================================== Exec ====================================
set(EXEC_TARGET_NAME Diffusion)

add_executable(${EXEC_TARGET_NAME} ../main.cpp
                                   $<TARGET_OBJECTS:${LIBRARY_NAME}>)

# Compile options and definitions
target_compile_definitions(${EXEC_TARGET_NAME}
                           PRIVATE ${BUILD_TYPE_DEFINITIONS})
target_compile_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})

# Linked libraries, linker options
target_link_libraries(${EXEC_TARGET_NAME} PRIVATE ${NESO_LIBRARY_NAME})
target_link_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_LINK_FLAGS})

# Enable sycl
add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES main.cpp)

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Exec rpath
set_target_properties(
  ${EXEC_TARGET_NAME} PROPERTIES INSTALL_RPATH ${INSTALL_RPATH}
                                 ${NEKTAR++_LIBRARY_DIRS})
