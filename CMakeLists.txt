cmake_minimum_required(VERSION 3.16)

project(CUDA_SIFT C CXX CUDA)

# Using C17 and C++17
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_STANDARD 14) set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile for all possible gpu compute capability cards
option(ALL_GPU_ARCHS OFF)
option(CUDASIFT_DETERMINISTIC_TESTING
       "Enable for reproducible random numbers in tests" OFF)
option(CUDASIFT_VISUAL_DEBUGGER "Create visual debugger program" OFF)

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

include(cmake/CPM.cmake)

if(NOT TARGET spdlog)
  message("spdlog not found while building cudasift - adding package")
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

if(CUDASIFT_VISUAL_DEBUGGER)
  set(BUILD_LIST
      "core,highgui,features2d,calib3d,flann"
      CACHE STRING " " FORCE)

  cpmaddpackage(
    NAME
    opencv
    GITHUB_REPOSITORY
    opencv/opencv
    # GIT_TAG 8.0.1 GIT_TAG 8.1.1
    GIT_TAG
    4.6.0
    EXCLUDE_FROM_ALL
    YES
    # EXCLUDE_FROM_ALL YES
    OPTIONS
    "BUILD_SHARED_LIBS Off"
    "BUILD_TESTS Off"
    "BUILD_EXAMPLES Off"
    "BUILD_PERF_TESTS Off"
    "BUILD_opencv_apps Off"
    "BUILD_TIFF On"
    "WITH_TIFF On"
    "WITH_PROTOBUF Off"
    # "BUILD_opencv_world Off" "OPENCV_EXTRA_MODULES_PATH
    # ${opencv_contrib_SOURCE_DIR}/modules" "WITH_CUDA On"
  )
endif()

# Set the source files for the cudasift library
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
  target_link_libraries(cudasift PUBLIC CUDA::cublas_static CUDA::cudart_static)
  if(OpenMP_C_FOUND)
    target_link_libraries(cudasift PUBLIC OpenMP::OpenMP_C)
  elseif(OpenMP_CXX_FOUND)
    target_link_libraries(cudasift PUBLIC OpenMP::OpenMP_CXX)
  endif()
else()
  target_link_libraries(cudasift PUBLIC CUDA::cublas_static CUDA::cudart_static)
  if(OpenMP_C_FOUND)
    target_link_libraries(cudasift PUBLIC OpenMP::OpenMP_C)
  elseif(OpenMP_CXX_FOUND)
    target_link_libraries(cudasift PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# ==================================
# Original Test Executable
# ==================================
if(WIN32)
  add_executable(test_cuda_sift cudasift_test.cu cuSIFT.cu geomFuncs_blas.cu
                                matching.cu cudaImage.cu cudaSift.cu)
  target_link_libraries(test_cuda_sift PUBLIC CUDA::cublas_static
                                              CUDA::cudart_static)
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

if(CUDASIFT_VISUAL_DEBUGGER)
  # ############################################################################
  # New Visual Debugger Executable - ADDED SECTION
  # ############################################################################
  if(WIN32)
    # On Windows, explicitly list sources similar to the test executable
    add_executable(
      visual_debugger visual_debugger.cpp cuSIFT.cu geomFuncs_blas.cu
                      matching.cu cudaImage.cu cudaSift.cu)
    target_link_libraries(
      visual_debugger PUBLIC CUDA::cublas_static CUDA::cudart_static
                             ${OpenCV_LIBS})
    if(OpenMP_C_FOUND)
      target_link_libraries(visual_debugger PUBLIC OpenMP::OpenMP_C)
    elseif(OpenMP_CXX_FOUND)
      target_link_libraries(visual_debugger PUBLIC OpenMP::OpenMP_CXX)
    endif()
  else()
    # On Unix, link against the static cudasift library
    add_executable(visual_debugger visual_debugger.cpp)
  endif()

  # Link common private libraries
  target_link_libraries(visual_debugger PRIVATE spdlog)
  target_link_libraries(visual_debugger PUBLIC cudasift
                                               ${OpenCVModules_TARGETS})
  target_include_directories(
    visual_debugger
    PUBLIC ${opencv_SOURCE_DIR}/include
           ${opencv_core_SOURCE_DIR}/include
           ${opencv_imgcodecs_SOURCE_DIR}/include
           ${opencv_imgproc_SOURCE_DIR}/include
           ${opencv_features2d_SOURCE_DIR}/include
           ${opencv_highgui_SOURCE_DIR}/include
           ${opencv_calib3d_SOURCE_DIR}/include
           ${opencv_flann_SOURCE_DIR}/include)
  # ############################################################################

  if(CUDASIFT_DETERMINISTIC_TESTING)
    target_compile_definitions(visual_debugger PRIVATE DETERMINISTIC_TESTING)
  endif()
endif()

if(CUDASIFT_DETERMINISTIC_TESTING)
  message(STATUS "CudaSift: Compiling with deterministic testing enabled.")
  target_compile_definitions(cudasift PRIVATE DETERMINISTIC_TESTING)
  target_compile_definitions(test_cuda_sift PRIVATE DETERMINISTIC_TESTING)
endif()

# Set all major architectures
if(${ALL_GPU_ARCHS})
  message("-- Building all gpu archs possible")
  set_property(TARGET cudasift PROPERTY CUDA_ARCHITECTURES all)
  set_property(TARGET test_cuda_sift PROPERTY CUDA_ARCHITECTURES all)
  # ADDED for the new executable
  set_property(TARGET visual_debugger PROPERTY CUDA_ARCHITECTURES all)
else()
  message("-- Building for currrent machine gpu arch only")
endif()

# Install
if(WIN32)
  install(
    TARGETS cudasift
    EXPORT "cudasiftExports"
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  install(
    TARGETS test_cuda_sift
    EXPORT "test_cuda_siftExports"
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  # ADDED for the new executable
  if(CUDASIFT_VISUAL_DEBUGGER)
    install(
      TARGETS visual_debugger
      EXPORT "visual_debuggerExports"
      RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endif()
  install(FILES cuSIFT.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
else()
  install(
    TARGETS cudasift
    EXPORT "cudasiftExports"
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  install(
    TARGETS test_cuda_sift
    EXPORT "test_cuda_siftExports"
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  # ADDED for the new executable
  if(CUDASIFT_VISUAL_DEBUGGER)
    install(
      TARGETS visual_debugger
      EXPORT "visual_debuggerExports"
      RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endif()
  install(FILES cuSIFT.h EXPORT "${PROJECT_NAME}Exports"
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()

# --- Add this block to create the parameter tuner executable ---
option(CUDASIFT_BUILD_TUNER "Build the parameter tuning utility" OFF)

if(CUDASIFT_BUILD_TUNER)
  message(STATUS "Building the parameter tuning utility.")

  # The tuner needs OpenCV for image loading.
  if(NOT TARGET opencv)
    set(BUILD_LIST
        "core,imgproc,imgcodecs,features2d"
        CACHE STRING " " FORCE)
    cpmaddpackage(
      NAME
      opencv
      GITHUB_REPOSITORY
      opencv/opencv
      GIT_TAG
      4.6.0
      EXCLUDE_FROM_ALL
      YES
      OPTIONS
      "BUILD_SHARED_LIBS Off"
      "BUILD_TESTS Off"
      "BUILD_EXAMPLES Off"
      "BUILD_PERF_TESTS Off"
      "BUILD_opencv_apps Off"
      "BUILD_TIFF On"
      "WITH_TIFF On"
      "WITH_PROTOBUF Off")
  endif()

  add_executable(param_tuner param_tuner.cpp)

  # Link against the cudasift library and other dependencies
  target_link_libraries(param_tuner PRIVATE cudasift spdlog)

  # Link OpenCV
  target_link_libraries(param_tuner PUBLIC ${OpenCVModules_TARGETS})
  target_include_directories(
    param_tuner
    PUBLIC ${opencv_SOURCE_DIR}/include
           ${opencv_core_SOURCE_DIR}/include
           ${opencv_imgcodecs_SOURCE_DIR}/include
           ${opencv_imgproc_SOURCE_DIR}/include
           ${opencv_features2d_SOURCE_DIR}/include
           ${opencv_highgui_SOURCE_DIR}/include
           ${opencv_calib3d_SOURCE_DIR}/include
           ${opencv_flann_SOURCE_DIR}/include)

  # Ensure it gets the DETERMINISTIC_TESTING flag if set
  if(CUDASIFT_DETERMINISTIC_TESTING)
    target_compile_definitions(param_tuner PRIVATE DETERMINISTIC_TESTING)
  endif()

  # Set GPU architectures
  if(${ALL_GPU_ARCHS})
    set_property(TARGET param_tuner PROPERTY CUDA_ARCHITECTURES all)
  endif()

  # Install the new executable
  install(
    TARGETS param_tuner
    EXPORT "param_tunerExports"
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

endif()
# --- End of new block ---
