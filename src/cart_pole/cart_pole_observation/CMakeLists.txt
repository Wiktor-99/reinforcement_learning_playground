cmake_minimum_required(VERSION 3.8)
project(cart_pole_observation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(cart_pole_observation_node src/cart_pole_observation_node.cpp)
target_include_directories(cart_pole_observation_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(cart_pole_observation_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cart_pole_observation_node
  "rclcpp"
  "sensor_msgs"
)

install(TARGETS cart_pole_observation_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
