from setuptools import find_packages, setup
from generate_parameter_library_py.setup_helper import generate_parameter_module

package_name = "cart_pole_reinforcement_learning"
generate_parameter_module(
    "cart_pole_reinforcement_learning_params",
    "cart_pole_reinforcement_learning/reinforcement_learning_node_parameters.yaml",
)

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Wiktor Bajor",
    maintainer_email="wiktorbajor1@gmail.com",
    description="Reinforcement node and simple algorithms.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cart_pole_basic_policy_node = cart_pole_reinforcement_learning.cart_pole_basic_policy_node:main",
            "cart_pole_neural_network_policy_node = cart_pole_reinforcement_learning.cart_pole_neural_network_policy:main",
            "cart_pole_deep_q_learning_policy_node = cart_pole_reinforcement_learning.cart_pole_deep_q_learning_policy_node:main",
        ],
    },
)
