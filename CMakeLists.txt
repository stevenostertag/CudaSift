cmake_minimum_required(VERSION 2.6)

project(cudasift LANGUAGES CUDA CXX)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include(GNUInstallDirs)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

option(BUILD_SHARED_LIB "Compile as a shared library" ON) #OFF by default

if(BUILD_SHARED_LIB)
  set(OPENCV_SHARED_LIBS "BUILD_SHARED_LIBS On")
else()
set(OPENCV_SHARED_LIBS "BUILD_SHARED_LIBS Off")
endif(BUILD_SHARED_LIB)

include(cmake/CPM.cmake)
set(BUILD_LIST "core,imgproc,imgcodecs,highgui" CACHE STRING " " FORCE)
CPMAddPackage(
    NAME opencv
    GITHUB_REPOSITORY opencv/opencv
    #GIT_TAG 8.0.1
    #GIT_TAG 8.1.1
    GIT_TAG 4.6.0
    # EXCLUDE_FROM_ALL YES
    OPTIONS
      ${OPENCV_SHARED_LIBS}
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

message(STATUS "OPENCV_CONFIG_FILE_INCLUDE_DIR=${OPENCV_CONFIG_FILE_INCLUDE_DIR}")
message(STATUS "OPENCV_MODULE_opencv_core_LOCATION=${OPENCV_MODULE_opencv_core_LOCATION}")

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  # list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
  list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE")
    # list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
    list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
  endif()
endif()

set(CUDASIFT_PUBLIC_INCLUDE_DIRECTORIES      ${CMAKE_CURRENT_SOURCE_DIR}/include/ )
set(CUDASIFT_PRIVATE_INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_SOURCE_DIR}/src/   )

file(GLOB_RECURSE CUDASIFT_SOURCE_FILES "src/*.cpp" "src/*.cu" "src/*.h")
file(GLOB_RECURSE CUDASIFT_INCLUDE_FILES "include/*.h")

# set(cuda_sources
#   src/cudaImage.cu  
#   src/cudaImage.h  
#   src/cudaSiftH.cu 
#   src/cudaSiftH.h  
#   src/matching.cu  
#   src/cudaSiftD.h  
#   include/cudaSift.h  
#   src/cudautils.h
# )  

set(sources
  src/geomFuncs.cpp  
  # mainSift.cpp  
  src/cudaSift.cpp
)

# include_directories(
#   ${CMAKE_CURRENT_SOURCE_DIR} 
# )

set(CUDASIFT_PUBLIC_HEADERS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/cudaSift.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/cudaImage.h>
)

#SET(CUDA_SEPARABLE_COMPILATION ON)
# cuda_add_library(${PROJECT_NAME} SHARED ${CUDASIFT_SOURCE_FILES} ${CUDASIFT_INCLUDE_FILES} OPTIONS -arch=sm_35)
if(BUILD_SHARED_LIB)
  add_library(${PROJECT_NAME} SHARED)
else()
  add_library(${PROJECT_NAME} STATIC)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_sources(${PROJECT_NAME}
PRIVATE 
${CUDASIFT_SOURCE_FILES} ${CUDASIFT_INCLUDE_FILES} 
PUBLIC
${CUDASIFT_PUBLIC_HEADERS})

target_sources(${PROJECT_NAME} PUBLIC FILE_SET HEADERS
  BASE_DIRS ${PROJECT_SOURCE_DIR}/include
  FILES ${CUDASIFT_INCLUDE_FILES}
)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # Contains export header
$<BUILD_INTERFACE:${CUDASIFT_PUBLIC_INCLUDE_DIRECTORIES}>)

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDASIFT_PRIVATE_INCLUDE_DIRECTORIES})

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    PUBLIC_HEADER "${CUDASIFT_INCLUDE_FILES}"
)

#cuda_add_executable(l2net l2netD.cu OPTIONS -arch=sm_35)

set_target_properties(${PROJECT_NAME} PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME}
  # ${CUDA_cudadevrt_LIBRARY} 
  ${OpenCVModules_TARGETS}
  # ${opencv_BINARY_DIR}/3rdparty/lib/libippiw.a
  Threads::Threads
)
target_include_directories(${PROJECT_NAME} PRIVATE 
  # ${CMAKE_CURRENT_BINARY_DIR}
  ${OPENCV_MODULE_opencv_core_LOCATION}/include 
  ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
  ${OPENCV_MODULE_opencv_highgui_LOCATION}/include
  ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include
  ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include
)
#  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS} 
#)

add_dependencies(${PROJECT_NAME} opencv_core opencv_imgproc opencv_imgcodecs)# ippiw)
 
# install(FILES 
#   ${cuda_sources} 
#   ${sources}
#   cudaSiftD.cu
#   CMakeLists.txt
#   # Copyright.txt
#   DESTINATION .
# )
# install(FILES data/left.pgm data/righ.pgm
#   DESTINATION data
# )

# install(TARGETS ${PROJECT_NAME})
install (TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Exports"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILE_SET HEADERS
)

install(EXPORT "${PROJECT_NAME}Exports"
    FILE ${PROJECT_NAME}Exports.cmake
    NAMESPACE cudasift::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudasift
)

# add_executable(${PROJECT_NAME}Exe mainSift.cpp)
# target_link_libraries(${PROJECT_NAME}Exe PUBLIC ${PROJECT_NAME})
# # target_include_directories(${PROJECT_NAME}Exe ${PROJECT_NAME})

# install (TARGETS ${PROJECT_NAME}Exe
#     RUNTIME DESTINATION bin
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     PUBLIC_HEADER DESTINATION include
# )
