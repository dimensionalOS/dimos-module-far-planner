cmake_minimum_required(VERSION 3.14)
project(far_planner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include(FetchContent)
FetchContent_Declare(dimos_lcm
  GIT_REPOSITORY https://github.com/dimensionalOS/dimos-lcm.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dimos_lcm)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LCM REQUIRED lcm)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common filters kdtree)
add_definitions(-DUSE_PCL)

find_package(OpenCV QUIET COMPONENTS core imgproc)

if(NOT DEFINED SMARTNAV_COMMON_DIR)
  set(SMARTNAV_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
endif()

add_executable(far_planner main.cpp)
target_include_directories(far_planner PRIVATE
  ${SMARTNAV_COMMON_DIR}
  ${dimos_lcm_SOURCE_DIR}/generated/cpp_lcm_msgs
  ${LCM_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(far_planner PRIVATE ${LCM_LIBRARIES} ${PCL_LIBRARIES})
target_link_directories(far_planner PRIVATE ${LCM_LIBRARY_DIRS})

if(OpenCV_FOUND)
  target_include_directories(far_planner PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(far_planner PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(far_planner PRIVATE HAS_OPENCV)
endif()

install(TARGETS far_planner DESTINATION bin)
