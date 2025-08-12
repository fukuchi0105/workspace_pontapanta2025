cmake_minimum_required(VERSION 3.8)
project(trajectory_follower_nobuakif)

# ----------------------------------------------------------------------------
# 1. Use ament_cmake_auto (drop autoware_cmake & autoware_package)
# ----------------------------------------------------------------------------
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# ----------------------------------------------------------------------------
# 4. Build your trajectory_follower_nobuakif node
# ----------------------------------------------------------------------------
ament_auto_add_executable(trajectory_follower_nobuakif_node
  src/trajectory_follower_nobuakif.cpp
)

# ----------------------------------------------------------------------------
# 5. Finalize package setup
# ----------------------------------------------------------------------------
ament_auto_package()

