cmake_minimum_required(VERSION 2.6)

project(cudaSift)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

include(cmake/CPM.cmake)
set(BUILD_LIST "core,imgproc,imgcodecs" CACHE STRING " " FORCE)
CPMAddPackage(
    NAME opencv
    GITHUB_REPOSITORY opencv/opencv
    #GIT_TAG 8.0.1
    #GIT_TAG 8.1.1
    GIT_TAG 4.6.0
    # EXCLUDE_FROM_ALL YES
    OPTIONS
      "BUILD_SHARED_LIBS On"
      "BUILD_TESTS Off"
      "BUILD_EXAMPLES Off"
      "BUILD_PERF_TESTS Off"
      "BUILD_opencv_apps Off"
)

# include(FetchContent)
# FetchContent_Declare(
#         opencv
#         GIT_REPOSITORY https://github.com/opencv/opencv.git
#         GIT_TAG 4.6.0
#         GIT_SHALLOW TRUE
#         GIT_PROGRESS TRUE
# )
# FetchContent_MakeAvailable(opencv)

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    if (ARGV0)
        unset(MATCHED)
        string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
        if (NOT MATCHED)
            continue()
        endif()
    endif()
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

message(STATUS "OPENCV_CONFIG_FILE_INCLUDE_DIR=${OPENCV_CONFIG_FILE_INCLUDE_DIR}")
message(STATUS "OPENCV_MODULE_opencv_core_LOCATION=${OPENCV_MODULE_opencv_core_LOCATION}")

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
  endif()
endif()

set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)  

set(sources
  geomFuncs.cpp  
  mainSift.cpp  
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_35)

#cuda_add_executable(l2net l2netD.cu OPTIONS -arch=sm_35)

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

find_package(Threads REQUIRED)

target_link_libraries(cudasift
  ${CUDA_cudadevrt_LIBRARY} 
  ${opencv_BINARY_DIR}/lib/libopencv_core.so
  ${opencv_BINARY_DIR}/lib/libopencv_imgproc.so
  ${opencv_BINARY_DIR}/lib/libopencv_imgcodecs.so
  pthread
)
target_include_directories(cudasift PRIVATE 
  ${OPENCV_MODULE_opencv_core_LOCATION}/include 
  ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
  ${OPENCV_MODULE_opencv_highgui_LOCATION}/include
  ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include
  ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include
)
#  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS} 
#)

add_dependencies(cudasift opencv_core opencv_imgproc opencv_imgcodecs)
 
install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)
