#include <rclcpp/rclcpp.hpp>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/entity.pb.h>
#include <gz/msgs/entity_factory.pb.h>
#include <gz/transport/Node.hh>
#include <std_srvs/srv/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <chrono>
#include <optional>

std::optional<std::string> get_robot_description_from_topic()
{
  std::string topic_name{"robot_description"};
  std::promise<std::string> robot_description_promise;
  std::shared_future<std::string> robot_description_future(robot_description_promise.get_future());
  rclcpp::executors::SingleThreadedExecutor executor;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr description_subs;
  auto ros2_node = std::make_shared<rclcpp::Node>("robot_description_acquire_node");
  executor.add_node(ros2_node);
  description_subs = ros2_node->create_subscription<std_msgs::msg::String>(topic_name, rclcpp::QoS(1).transient_local(),
    [&robot_description_promise](const std_msgs::msg::String::SharedPtr msg) {
      robot_description_promise.set_value(msg->data);
    });

  rclcpp::FutureReturnCode future_ret;
  do
  {
    RCLCPP_INFO(ros2_node->get_logger(), "Waiting messages on topic [%s].", topic_name.c_str());
    future_ret = executor.spin_until_future_complete(robot_description_future, std::chrono::seconds(1));
  } while (rclcpp::ok() && future_ret != rclcpp::FutureReturnCode::SUCCESS);

  if (future_ret != rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_ERROR(
      ros2_node->get_logger(), "Failed to get XML from topic [%s].", topic_name.c_str());
    return std::nullopt;
  }
  return robot_description_future.get();
}

class SimulationControlNode : public rclcpp::Node
{
public:
  SimulationControlNode(const rclcpp::NodeOptions& options)
  : rclcpp::Node("simulation_control_node", options)
  {
    robot_description_ = *get_robot_description_from_topic();
    server_ = create_service<std_srvs::srv::Empty>("restart_sim_service", [&](
      std_srvs::srv::Empty::Request::SharedPtr req,
      std_srvs::srv::Empty::Response::SharedPtr resp)
      {
        restart_sim(req, resp);
      });
  }

  void restart_sim(std_srvs::srv::Empty::Request::SharedPtr, std_srvs::srv::Empty::Response::SharedPtr)
  {
    delete_cart_pole();
    create_cart_pole();
  }

  void delete_cart_pole()
  {
    gz::transport::Node node;
    gz::msgs::Entity req;
    req.set_name(robot_to_remove_name_);
    req.set_type(gz::msgs::Entity_Type_MODEL);
    gz::msgs::Boolean rep;
    bool result;

    while(rclcpp::ok() and not node.Request(service_remove_, req, timeout_, rep, result))
    {
      RCLCPP_WARN(
        this->get_logger(), "Waiting for service [%s] to become available ...", service_remove_.c_str());
    }
  }

  void create_cart_pole()
  {
    gz::transport::Node node;
    bool result;
    gz::msgs::EntityFactory robot_spawn_req;
    gz::msgs::Boolean rep;
    robot_spawn_req.set_sdf(robot_description_);
    robot_spawn_req.set_name(robot_name_);

    while(rclcpp::ok() and not node.Request(service_create_, robot_spawn_req, timeout_, rep, result))
    {
      RCLCPP_WARN(
        this->get_logger(), "Waiting for service [%s] to become available ...", service_create_.c_str());
    }
    update_robot_name();
  }

  void update_robot_name()
  {
    counter_ += 1;
    robot_to_remove_name_ = robot_name_;
    robot_name_ = std::string("cart_pole") + std::to_string(counter_);
  }

  int counter_{};
  std::string robot_to_remove_name_{"cart_pole"};
  std::string robot_name_{"cart_pole"};
  std::string robot_description_{};
  const unsigned int timeout_{5000};
  const std::string service_create_{"/world/empty/create"};
  const std::string service_remove_{"/world/empty/remove"};
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr server_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(SimulationControlNode)