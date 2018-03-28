cmake_minimum_required(VERSION 2.8)
project(FOOL)

# Add definitions
add_definitions(-DCPU_ONLY)

set(LINKER_LIBS "")

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Wno-unused-variable -Wsign-compare -pthread")

# opencv
set(OpenCV_DIR /opt/opencv-2.4.9/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${OpenCV_LIBS})

# glog
include("cmake/FindGlog.cmake")
#find_package(lib_glog REQUIRED)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${GLOG_LIBRARIES})

# gflags
include("cmake/FindGFlags.cmake")
#find_package(GFlags REQUIRED)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${GFLAGS_LIBRARIES})

#FILE(GLOB SRC_FILES "*.cpp")
#FOREACH(FILE_PATH ${SRC_FILES})
#	MESSAGE(${FILE_PATH})
#	STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${FILE_PATH})
#	MESSAGE(${FILE_NAME})
# ENDFOREACH(FILE_PATH)

# first compile then link
add_executable(${PROJECT_NAME} "main.cpp" "block.cpp" "memory_controller.cpp" "filler_filter.cpp" "fully_connect_filter.cpp")
#	"io.cpp" "flatten_filter.cpp" "data_layer.cpp")
#add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${LINKER_LIBS})

# ${GTEST_DIR}/src/gtest-all.cc


