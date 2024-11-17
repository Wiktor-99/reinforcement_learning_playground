from setuptools import find_packages, setup

package_name = 'cart_pole_model_based_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jakub Delicat',
    maintainer_email='delicat.kuba@gmail.com',
    description='Cart pole dynamic model based controller',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "check_model = cart_pole_model_based_controller.check_model:main",
            "linear_approximation_control_node = cart_pole_model_based_controller.linear_approximation_control_node:main",
            "dynamics = cart_pole_model_based_controller.dynamics:main",
        ],
    },
)
