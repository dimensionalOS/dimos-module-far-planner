cmake_minimum_required(VERSION 3.14)
project(far_planner_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/result" CACHE PATH "" FORCE)
endif()

# OpenMP for PCL parallel processing
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Fetch dimos-lcm C++ message headers
include(FetchContent)
FetchContent_Declare(dimos_lcm
  GIT_REPOSITORY https://github.com/dimensionalOS/dimos-lcm.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dimos_lcm)

# LCM
find_package(PkgConfig REQUIRED)
pkg_check_modules(LCM REQUIRED lcm)

# Eigen3
find_package(Eigen3 REQUIRED)

# PCL (common + filters + search + kdtree)
find_package(PCL 1.8 REQUIRED COMPONENTS common filters search kdtree)

# OpenCV (for contour detection)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

# Boost (for hash in node_struct.h)
find_package(Boost REQUIRED)

# SmartNav common directory (dimos_native_module.hpp, point_cloud_utils.hpp)
if(NOT SMART_NAV_COMMON_DIR)
  set(SMART_NAV_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
endif()

add_executable(far_planner_native
  main.cpp
  src/utility.cpp
  src/dynamic_graph.cpp
  src/contour_detector.cpp
  src/contour_graph.cpp
  src/graph_planner.cpp
  src/map_handler.cpp
  src/scan_handler.cpp
  src/terrain_planner.cpp
  src/graph_msger.cpp
  src/planner_visualizer.cpp
)

target_include_directories(far_planner_native PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${dimos_lcm_SOURCE_DIR}/generated/cpp_lcm_msgs
  ${LCM_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${SMART_NAV_COMMON_DIR}
)

target_link_libraries(far_planner_native PRIVATE
  ${LCM_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(far_planner_native PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_directories(far_planner_native PRIVATE
  ${LCM_LIBRARY_DIRS}
)

install(TARGETS far_planner_native DESTINATION bin)
