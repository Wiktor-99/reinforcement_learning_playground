import rclpy
from rclpy.node import Node
from cart_pole_observation_interface.msg import CartPoleObservation
from std_msgs.msg import Float64
import numpy as np
from std_srvs.srv import Empty
import control as ctrl


class LinearApproximationControlNode(Node):
    def __init__(self, node_name="linear_approximation_control_node"):
        super().__init__(node_name)
        self.observation_subscriber = self.create_subscription(
            CartPoleObservation, "observations", self.store_observation, 10
        )
        self.simulation_reset_service_client = self.create_client(
            Empty, "restart_sim_service"
        )
        self.effort_command_publisher = self.create_publisher(Float64, "effort_cmd", 10)
        self.cart_observations = np.zeros(4) # x, dx , theta, dtheta
        self.is_truncated = False
        self.UPPER_POLE_LIMIT = 1.4
        self.LOWER_POLE_LIMIT = -1.4
        
        
        m = 0.455  # Mass of the pendulum [kg]
        M = 1.006  # Mass of the cart
        l = 0.435057  # Length to the pendulum's center of mass [m]
        Ip = 0.044096

        # Given system matrices (A, B)
        A = self.get_A(l, m, M, Ip)
        B = self.get_B(l, m, M, Ip)
        self.K = ctrl.place(A, B, [-1, -2, -3, -4])  # Place the poles of the system
        self.C = np.array([0, 1, 0, 1])

        self.timer = self.create_timer(0.1, self.update)

    def store_observation(self, cart_pole_observation: CartPoleObservation):
        self.cart_observations[0] = cart_pole_observation.cart_position
        self.cart_observations[1] = cart_pole_observation.pole_angle
        self.cart_observations[2] = cart_pole_observation.cart_velocity
        self.cart_observations[3] = cart_pole_observation.pole_angular_velocity
        self.update_simulation_status()

    def get_cart_observations(self):
        return self.cart_observations

    def is_simulation_stopped(self):
        return self.is_truncated

    def take_action(self, action):
        self.effort_command_publisher.publish(action)

    def reset_observation(self):
        self.cart_observations = [0.0, 0.0, 0.0, 0.0]
        self.is_truncated = False

    def update_simulation_status(self):
        self.is_truncated = np.isclose(
            self.cart_observations[1],
            self.UPPER_POLE_LIMIT,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=False,
        ) or np.isclose(
            self.cart_observations[1],
            self.LOWER_POLE_LIMIT,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=False,
        )

    def restart_simulation(self):
        while not self.simulation_reset_service_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                "restart_sim_service not available, waiting again..."
            )

        # FIXME: future callback inside timer callback
        future = self.simulation_reset_service_client.call_async(Empty.Request())
        # rclpy.spin_until_future_complete(self, future)

    def get_A(self, l, m, M, Ip):
        g = 9.81
        return np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, g * l**2 * m**2 / (-Ip * M - Ip * m - M * l**2 * m), 0, 0],
                [0, g * l * m * (-M - m) / (-Ip * M - Ip * m - M * l**2 * m), 0, 0],
            ]
        )

    def get_B(self, l, m, M, Ip):
        return np.array(
            [
                [0],
                [0],
                [(-Ip - l**2 * m) / (-Ip * M - Ip * m - M * l**2 * m)],
                [l * m / (-Ip * M - Ip * m - M * l**2 * m)],
            ]
        )

    def control(self, x):
        y = np.dot(self.K * self.C, x)[0]
        print(f"x: {x}")
        print(f"y: {y}")

        return -y

    def update(self):
        self.update_simulation_status()
        if self.is_simulation_stopped():
            self.reset_observation()
            self.get_logger().info("Simulation stopped. Restarting...")
            self.restart_simulation()
            return

        y = self.control(self.get_cart_observations())
        self.take_action(Float64(data=y))


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = LinearApproximationControlNode()
    rclpy.spin(cart_pole_reinforcement_learning)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
