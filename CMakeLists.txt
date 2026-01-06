cmake_minimum_required(VERSION 3.16)

project(CUDA_SIFT C CXX CUDA)

# Using C17 and C++17 set(CMAKE_C_STANDARD 14) set(CMAKE_C_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_STANDARD 14) set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile for all possible gpu compute capability cards
option(ALL_GPU_ARCHS OFF)

# if(WIN32) # Make sure to statically link C/C++ windows runtime libs
# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# endif()

# Find Cuda Toolkit
find_package(CUDAToolkit REQUIRED)

# Find optional openmp
find_package(OpenMP)

# spdlog needs to be built with fpic, jumping through some hoops to prevent
# affecting top level cmake
if(DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(FORCE_FPIC ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE_OLD ${CMAKE_POSITION_INDEPENDENT_CODE})
else()
  set(FORCE_FPIC OFF)
endif()

if(NOT TARGET spdlog)

  message("spdlog not found while building cudasift - adding package")

  include(cmake/CPM.cmake)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  cpmaddpackage(
    NAME
    spdlog
    GITHUB_REPOSITORY
    gabime/spdlog
    GIT_TAG
    v1.13.0
    OPTIONS
    "SPDLOG_INSTALL ON")

  if(FORCE_FPIC)
    set(CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE_OLD})
  else()
    unset(CMAKE_POSITION_INDEPENDENT_CODE)
  endif()

endif()

# Set the source files
set(SRC cuSIFT.cu geomFuncs_blas.cu matching.cu cudaImage.cu cudaSift.cu)

set(CUDASIFT_PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/)

if(WIN32)
  add_library(cudasift STATIC ${SRC})
elseif(UNIX)
  add_library(cudasift STATIC ${SRC})
else()
  message(FATAL_ERROR "Only Windows and Linux supported")
endif()

target_include_directories(
  cudasift
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # Contains export header
         $<BUILD_INTERFACE:${CUDASIFT_PUBLIC_INCLUDE_DIRECTORIES}>)

target_link_libraries(cudasift PRIVATE spdlog)

# Link libraries, only cuda
if(WIN32)
  target_link_libraries(cudasift PUBLIC CUDA::cublas CUDA::cudart_static)
  if(OpenMP_C_FOUND)
    target_link_libraries(cudasift PUBLIC OpenMP::OpenMP_C)
  elseif(OpenMP_CXX_FOUND)
    target_link_libraries(cudasift PUBLIC OpenMP::OpenMP_CXX)
  endif()
else()
  target_link_libraries(cudasift PUBLIC CUDA::cublas CUDA::cudart_static)
  if(OpenMP_C_FOUND)
    target_link_libraries(cudasift PUBLIC OpenMP::OpenMP_C)
  elseif(OpenMP_CXX_FOUND)
    target_link_libraries(cudasift PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(WIN32)
  add_executable(test_cuda_sift cudasift_test.cu cuSIFT.cu geomFuncs_blas.cu
                                matching.cu cudaImage.cu cudaSift.cu)
  target_link_libraries(test_cuda_sift PUBLIC CUDA::cublas CUDA::cudart_static)
  if(OpenMP_C_FOUND)
    target_link_libraries(test_cuda_sift PUBLIC OpenMP::OpenMP_C)
  elseif(OpenMP_CXX_FOUND)
    target_link_libraries(test_cuda_sift PUBLIC OpenMP::OpenMP_CXX)
  endif()
else()
  add_executable(test_cuda_sift cudasift_test.cu)
  target_link_libraries(test_cuda_sift PUBLIC cudasift)

  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
endif()

target_link_libraries(test_cuda_sift PRIVATE spdlog)

# Set all major architectures
if(${ALL_GPU_ARCHS})
  message("-- Building all gpu archs possible")
  set_property(TARGET cudasift PROPERTY CUDA_ARCHITECTURES all)
  set_property(TARGET test_cuda_sift PROPERTY CUDA_ARCHITECTURES all)
else()
  message("-- Building for currrent machine gpu arch only")
endif()

# Install if(WIN32) install( TARGETS cudasift EXPORT "cudasiftExports" RUNTIME
# DESTINATION ${CMAKE_INSTALL_PREFIX}/bin) install( TARGETS test_cuda_sift
# EXPORT "test_cuda_siftExports" RUNTIME DESTINATION
# ${CMAKE_INSTALL_PREFIX}/bin) install(FILES cuSIFT.h DESTINATION
# ${CMAKE_INSTALL_PREFIX}/include) else() install( TARGETS cudasift EXPORT
# "cudasiftExports" LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib) install(
# TARGETS test_cuda_sift EXPORT "test_cuda_siftExports" RUNTIME DESTINATION
# ${CMAKE_INSTALL_PREFIX}/bin) install(FILES cuSIFT.h EXPORT
# "${PROJECT_NAME}Exports" DESTINATION ${CMAKE_INSTALL_PREFIX}/include) endif()

# install( EXPORT "test_cuda_siftExports" FILE test_cuda_siftExports.cmake
# NAMESPACE cudasift:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudasift)
#
# install( EXPORT "cudasiftExports" FILE cudasiftExports.cmake NAMESPACE
# cudasift:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudasift)
