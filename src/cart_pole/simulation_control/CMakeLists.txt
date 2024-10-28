cmake_minimum_required(VERSION 3.8)
project(simulation_control)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rclcpp_components REQUIRED)

find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport REQUIRED)

find_package(gz_msgs_vendor REQUIRED)
find_package(gz-msgs REQUIRED)

set(GZ_SIM_VER ${gz-sim_VERSION_MAJOR})

add_library(simulation_control SHARED src/simulation_control_node.cpp)
ament_target_dependencies(simulation_control rclcpp rclcpp_components std_srvs)
target_link_libraries(simulation_control gz-msgs::core gz-transport::core)
rclcpp_components_register_node(simulation_control PLUGIN "SimulationControlNode" EXECUTABLE simulation_control_node)

install(TARGETS
simulation_control
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_package()
