import rclpy
import time
from std_msgs.msg import Float64
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning
import tensorflow as tf
import copy
import numpy as np


class CartPoleReinforcementBasicPolicy:
    def __init__(self):
        self.MAX_STEPS_IN_EPISODE = 500
        self.MAX_ITERATION = 150
        self.MAX_EPISODE_PER_UPDATE = 10
        self.MAX_EFFORT_COMMAND = 5.0
        self.TIME_BETWEEN_COMMANDS = 0.001
        self.learning_control_node = CartPoleReinforcementLearning("cart_pole_neural_network_node")
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

    def create_command(self, action):
        return Float64(data=self.MAX_EFFORT_COMMAND) if action == 0 else Float64(data=-self.MAX_EFFORT_COMMAND)

    def play_one_step(self, loss_fn):
        observations = np.array(copy.deepcopy(self.learning_control_node.get_cart_observations()))
        with tf.GradientTape() as tape:
            left_proba = self.model(observations[np.newaxis])
            action = tf.random.uniform([1, 1]) > left_proba
            y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
            loss = tf.reduce_mean(loss_fn(y_target, left_proba))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.learning_control_node.take_action(self.create_command(int(action)))
        return grads

    def play_multiple_episodes(self, loss_fn):
        all_rewards = []
        all_grads = []
        for _ in range(self.MAX_EPISODE_PER_UPDATE):
            current_rewards = []
            current_grads = []
            for _ in range(self.MAX_STEPS_IN_EPISODE):
                current_grads.append(self.play_one_step(loss_fn))
                current_rewards.append(1.0)
                if self.learning_control_node.is_simulation_stopped():
                    break
                rclpy.spin_once(self.learning_control_node)
                time.sleep(self.TIME_BETWEEN_COMMANDS)

            self.learning_control_node.restart_simulation()
            self.learning_control_node.reset_observation()

            all_rewards.append(current_rewards)
            all_grads.append(current_grads)

        return all_rewards, all_grads

    def discount_rewards(self, rewards, discount_factor):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_factor
        return discounted

    def discount_and_normalize_rewards(self, all_rewards, discount_factor):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_factor) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    def run_neural_network_policy(self):
        optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
        loss_fn = tf.keras.losses.binary_crossentropy
        discount_factor = 0.97

        for iteration in range(self.MAX_ITERATION):
            all_rewards, all_grads = self.play_multiple_episodes(loss_fn)
            all_final_rewards = self.discount_and_normalize_rewards(all_rewards, discount_factor)

            self.learning_control_node.get_logger().info(
                f"Ended {iteration} iteration with max final {max([sum(rewards) for rewards in all_rewards])}"
            )
            all_mean_grads = []
            for var_index in range(len(self.model.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [
                        final_reward * all_grads[episode_index][step][var_index]
                        for episode_index, final_rewards in enumerate(all_final_rewards)
                        for step, final_reward in enumerate(final_rewards)
                    ],
                    axis=0,
                )
                all_mean_grads.append(mean_grads)

            optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementBasicPolicy()
    cart_pole_reinforcement_learning.run_neural_network_policy()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
