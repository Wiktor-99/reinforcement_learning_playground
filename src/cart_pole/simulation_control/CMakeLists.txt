cmake_minimum_required(VERSION 3.8)
project(simulation_control)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(gz_msgs_vendor REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz_transport_vendor REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(simulation_control_node src/simulation_control_node.cpp)
target_include_directories(simulation_control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(simulation_control_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  simulation_control_node
  "rclcpp"
  "gz_msgs_vendor"
  "gz_sim_vendor"
  "gz_transport_vendor"
  "std_msgs"
)

install(TARGETS simulation_control_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
