cmake_minimum_required(VERSION 3.8)
project(trajectory_follower_nobuakif)

find_package(autoware_cmake REQUIRED)
autoware_package()

# find dependencies
find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_executable(trajectory_follower_nobuakif
  src/trajectory_follower_nobuakif.cpp
)

ament_auto_package()
