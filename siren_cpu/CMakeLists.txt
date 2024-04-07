add_definitions(-w)
add_catch(test_siren run.cpp)

set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

target_compile_definitions(test_siren PUBLIC BASE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

target_include_directories(test_siren PUBLIC ../siren_cpu/camera)
target_include_directories(test_siren PUBLIC ../siren_cpu/image_utils)
target_include_directories(test_siren PUBLIC ../siren_cpu/geometry)
target_include_directories(test_siren PUBLIC ../siren_cpu/postprocessing)
target_include_directories(test_siren PUBLIC ../siren_cpu/scene)
target_include_directories(test_siren PUBLIC ../siren_cpu/siren)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


target_link_libraries(test_siren ${PNG_LIBRARY} ${JPEG_LIBRARIES})
target_include_directories(
  test_siren
  PRIVATE ${PNG_INCLUDE_DIRS} ${JPEG_INCLUDE_DIRS}
)
