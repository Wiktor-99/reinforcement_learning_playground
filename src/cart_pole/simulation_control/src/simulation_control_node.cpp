#include <rclcpp/rclcpp.hpp>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/entity.pb.h>
#include <gz/msgs/entity_factory.pb.h>
#include <gz/transport/Node.hh>
#include <std_srvs/srv/empty.hpp>

class SimulationControlNode : public rclcpp::Node
{
public:
  SimulationControlNode(const rclcpp::NodeOptions& options)
  : rclcpp::Node("simulation_control_node", options)
  {
    server_ = create_service<std_srvs::srv::Empty>("restart_sim_service", [&](
      std_srvs::srv::Empty::Request::SharedPtr req,
      std_srvs::srv::Empty::Response::SharedPtr resp)
      {
        restart_sim(req, resp);
      });
  }


  void restart_sim(std_srvs::srv::Empty::Request::SharedPtr, std_srvs::srv::Empty::Response::SharedPtr)
  {

  }

  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr server_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(SimulationControlNode)