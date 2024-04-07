project(NAIR-V)

cmake_minimum_required(VERSION 3.8)


set(CMAKE_CXX_STANDARD             17)
set(CMAKE_MODULE_PATH              "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)

find_package(PNG)
find_package(JPEG)
find_package(Catch REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(cmake/TestSolution.cmake)
include(cmake/BuildFlags.cmake)


if (${PNG_FOUND} AND ${JPEG_FOUND})
  add_subdirectory(siren_cpu)
elseif()
  message(STATUS "siren_cpu disabled. PNG:${PNG_FOUND}")
endif()
