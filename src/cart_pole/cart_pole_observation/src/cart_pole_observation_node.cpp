#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "cart_pole_observation_interface/msg/cart_pole_observation.hpp"

using cart_pole_observation_interface::msg::CartPoleObservation;

class CartPoleObserver : public rclcpp::Node
{
public:
  CartPoleObserver() : Node("cart_pole_observer")
  {
    cart_pole_state_publisher_ = create_publisher<CartPoleObservation>("observations", 10);
    auto topic_callback = [this](sensor_msgs::msg::JointState::UniquePtr msg)
    {
      const auto slider_to_cart_iter = std::find(msg->name.cbegin(), msg->name.cend(), "slider_to_cart");
      const auto pole_holder_base_to_pole_holder_iter =
        std::find(msg->name.cbegin(), msg->name.cend(), "slider_to_pole_with_holder");
      CartPoleObservation cart_pole_observation;
      cart_pole_observation.cart_position = msg->position[std::distance(msg->name.cbegin(), slider_to_cart_iter)];
      cart_pole_observation.cart_velocity = msg->velocity[std::distance(msg->name.cbegin(), slider_to_cart_iter)];
      cart_pole_observation.pole_angle =
        msg->position[std::distance(msg->name.cbegin(), pole_holder_base_to_pole_holder_iter)];
      cart_pole_observation.pole_angular_velocity =
        msg->velocity[std::distance(msg->name.cbegin(), pole_holder_base_to_pole_holder_iter)];
      cart_pole_state_publisher_->publish(cart_pole_observation);
    };
    joint_states_subscriber_ = create_subscription<sensor_msgs::msg::JointState>("joint_states", 10, topic_callback);
  }

private:
  rclcpp::Publisher<CartPoleObservation>::SharedPtr cart_pole_state_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_subscriber_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CartPoleObserver>());
  rclcpp::shutdown();
  return 0;
}
