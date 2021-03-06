#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.1.0 )

project ( RTWeekend
  VERSION 3.0.0
  LANGUAGES CXX
)

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

set ( COMMON_ALL
  src/common/vec3.h
  src/common/color.h
  src/common/ray.h
  src/common/rtweekend.h
  src/common/camera.h
)

set ( SOURCE_ONE_WEEKEND
  ${COMMON_ALL}
  src/InOneWeekend/main.cc
  src/InOneWeekend/hittable.h
  src/InOneWeekend/sphere.h
  src/InOneWeekend/hittable_list.h
  src/InOneWeekend/material.h
)

# Executables
add_executable(inOneWeekend      ${SOURCE_ONE_WEEKEND})

include_directories(src/common)
