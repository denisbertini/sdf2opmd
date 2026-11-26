cmake_minimum_required(VERSION 3.22)
project(sdf2opmd VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(openPMD REQUIRED)

# ---------------------------------------------------------------------------
# SDF C library (epochpic/SDF_C), built with MPI-IO support (PARALLEL=ON).
# ---------------------------------------------------------------------------
set(PARALLEL ON CACHE BOOL "SDF_C parallel (MPI-IO) build" FORCE)
add_subdirectory(extern/sdf EXCLUDE_FROM_ALL)

add_executable(sdf2opmd
  src/main.cpp
  src/cli.cpp
  src/blockmap.cpp
  src/mapping.cpp
  src/convert.cpp
)
target_include_directories(sdf2opmd PRIVATE src third_party extern/sdf/include)
# sdf.h exposes MPI types only when PARALLEL is defined by the consumer too
target_compile_definitions(sdf2opmd PRIVATE PARALLEL
  SDF2OPMD_VERSION="${PROJECT_VERSION}")
target_compile_options(sdf2opmd PRIVATE -Wall -Wextra)
target_link_libraries(sdf2opmd PRIVATE openPMD::openPMD MPI::MPI_CXX sdfc)
# ADIOS2 core (bbstage build) links libfabric without propagating it
find_library(FABRIC_LIBRARY NAMES fabric)

# Test utility: reads back a converted openPMD series and validates it
add_executable(readback tests/readback.cpp)
target_link_libraries(readback PRIVATE openPMD::openPMD MPI::MPI_CXX)

# openPMD-only write pattern stress test (no SDF code)
add_executable(stress_openpmd tests/stress_openpmd.cpp)
add_executable(sdfdump tests/sdfdump.cpp)
target_include_directories(sdfdump PRIVATE src extern/sdf/include)
target_compile_definitions(sdfdump PRIVATE PARALLEL)
target_link_libraries(sdfdump PRIVATE MPI::MPI_CXX sdfc)
target_link_libraries(stress_openpmd PRIVATE openPMD::openPMD MPI::MPI_CXX)
if(FABRIC_LIBRARY)
  target_link_libraries(sdf2opmd PRIVATE "${FABRIC_LIBRARY}")
  target_link_libraries(readback PRIVATE "${FABRIC_LIBRARY}")
  target_link_libraries(stress_openpmd PRIVATE "${FABRIC_LIBRARY}")
endif()

enable_testing()
add_test(NAME fixture_roundtrip
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_e2e.sh
          $<TARGET_FILE:sdf2opmd> $<TARGET_FILE:readback>
          ${CMAKE_CURRENT_SOURCE_DIR})
