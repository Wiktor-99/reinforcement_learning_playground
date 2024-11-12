cmake_minimum_required(VERSION 3.8)
project(cart_pole_bringup)


find_package(ament_cmake REQUIRED)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.sh.in")

install(DIRECTORY
  launch
  config
  meshes
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
