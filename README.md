# Reinforcement learning playground

Currently repository allows to play with cart pole model.

### Cart pole simulation
![cart pole](/img/cart_pole.png " ")

### Model of the cart pole

The model of the cart pole was initially created and analyzed in publication:

* > K. Arent, M. Szulc,
  > **"Experimental stand with a pendulum on a cart: construction, modelling, identification, model based control and deployment"**

        @chapter {ARENT2018
            title = "Experimental stand with a pendulum on a cart: construction, modelling, identification, model based control and deployment"
            authors = "KRZYSZTOF ARENT and Mateusz Szulc"
            booktitle = "Postępy robotyki"
            year = "2018"
            pages = "365-376"
        }

The cart pole is located in [The Laboratory of Robotics](https://lr.kcir.pwr.edu.pl/) on [Wrocaław University of Science and Technology](https://pwr.edu.pl/).



### Run simulation

Open folder in a devcontainer. Then type

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch cart_pole_bringup cart_pole.launch.py
```

### Linear approximation in an unstable equilibrium point

A model of an inverted pendulum on a cart and controller are derived in the `cart_pole_system.ipynb`.

The model was applied to control the cart pole in the script `linear_approximation_control_node.py`.

```bash
source install/setup.bash
ros2 run cart_pole_model_based_controller linear_approximation_control_node
```

### Reinforcement learning nodes

Repository currently contains three reinforcement learning policies:
* Basic policy
* Neural network policy
* Deep q learning policy

All implementations are based on the implementations presented in the `Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow: Concepts, Tools, and Techniques to Build Intelligent`

### Launching
To run simulations use command given above and in other terminal use one of following command:

```bash
source install/setup.bash
ros2 run cart_pole_reinforcement_learning cart_pole_basic_policy_node
```

or

```bash
source install/setup.bash
ros2 run cart_pole_reinforcement_learning cart_pole_neural_network_policy
```

or

```bash
source install/setup.bash
ros2 run cart_pole_reinforcement_learning cart_pole_deep_q_learning_policy_node
```
