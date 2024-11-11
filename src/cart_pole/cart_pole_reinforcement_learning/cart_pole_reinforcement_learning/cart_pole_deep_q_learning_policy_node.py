import rclpy
import time
from std_msgs.msg import Float64
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning
import tensorflow as tf
import copy
import numpy as np
from collections import deque


class CartPoleReinforcementDeepQLearningPolicy:
    def __init__(self):
        self.MAX_EPISODES = 600
        self.MAX_STEPS = 200
        self.MAX_EFFORT_COMMAND = 10.0
        self.TIME_BETWEEN_COMMANDS = 0.02
        self.learning_control_node = CartPoleReinforcementLearning("cart_pole_deep_q_learning_policy_node")
        self.input_shape = [4]
        self.n_outputs = 2
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="elu", input_shape=self.input_shape),
                tf.keras.layers.Dense(32, activation="elu"),
                tf.keras.layers.Dense(self.n_outputs),
            ]
        )
        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 32
        self.discount_factor = 0.95
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
        self.loss_fn = tf.keras.losses.mse

    def create_command(self, action):
        return Float64(data=self.MAX_EFFORT_COMMAND) if action == 0 else Float64(data=-self.MAX_EFFORT_COMMAND)

    def epsilon_greedy_policy(self, epsilon, state):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
            return q_values.argmax()

    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        return [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]

    def play_one_step(self, epsilon):
        state = np.array(copy.deepcopy(self.learning_control_node.get_cart_observations()))
        action = self.epsilon_greedy_policy(epsilon, state)
        self.learning_control_node.take_action(self.create_command(action))
        next_state = np.array(copy.deepcopy(self.learning_control_node.get_cart_observations()))
        self.replay_buffer.append((state, action, 1, next_state, self.learning_control_node.is_simulation_stopped()))

    def training_step(self):
        states, actions, rewards, next_states, truncateds = self.sample_experiences()
        next_Q_values = self.model.predict(next_states, verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = 1.0 - truncateds
        target_Q_values = rewards + runs * self.discount_factor * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def run(self):
        for episode in range(self.MAX_EPISODES):
            reward = 0
            for _ in range(self.MAX_STEPS):
                epsilon = max(1 - episode / self.MAX_EPISODES, 0.01)
                self.play_one_step(epsilon)
                reward += 1
                if self.learning_control_node.is_simulation_stopped():
                    break

            if episode > 50:
                self.training_step()

            self.learning_control_node.restart_simulation()
            self.learning_control_node.reset_observation()
            self.learning_control_node.get_logger().info(f"Ended episode: {episode} with score: {reward}")


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementDeepQLearningPolicy()
    cart_pole_reinforcement_learning.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
