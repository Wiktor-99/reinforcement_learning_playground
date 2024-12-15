# flake8: noqa

# auto-generated DO NOT EDIT

from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.msg import FloatingPointRange, IntegerRange
from rclpy.clock import Clock
from rclpy.exceptions import InvalidParameterValueException
from rclpy.time import Time
import copy
import rclpy
import rclpy.parameter
from generate_parameter_library_py.python_validators import ParameterValidators


class reinforcement_learning_node_parameters:

    class Params:
        # for detecting if the parameter struct has been updated
        stamp_ = Time()

        max_number_of_episodes = 100
        max_number_of_steps = 500
        max_effort_command = 5.0
        discount_factor = 0.95
        reward = 1

    class ParamListener:
        def __init__(self, node, prefix=""):
            self.prefix_ = prefix
            self.params_ = reinforcement_learning_node_parameters.Params()
            self.node_ = node
            self.logger_ = rclpy.logging.get_logger("reinforcement_learning_node_parameters." + prefix)

            self.declare_params()

            self.node_.add_on_set_parameters_callback(self.update)
            self.clock_ = Clock()

        def get_params(self):
            tmp = self.params_.stamp_
            self.params_.stamp_ = None
            paramCopy = copy.deepcopy(self.params_)
            paramCopy.stamp_ = tmp
            self.params_.stamp_ = tmp
            return paramCopy

        def is_old(self, other_param):
            return self.params_.stamp_ != other_param.stamp_

        def unpack_parameter_dict(self, namespace: str, parameter_dict: dict):
            """
            Flatten a parameter dictionary recursively.

            :param namespace: The namespace to prepend to the parameter names.
            :param parameter_dict: A dictionary of parameters keyed by the parameter names
            :return: A list of rclpy Parameter objects
            """
            parameters = []
            for param_name, param_value in parameter_dict.items():
                full_param_name = namespace + param_name
                # Unroll nested parameters
                if isinstance(param_value, dict):
                    nested_params = self.unpack_parameter_dict(
                        namespace=full_param_name + rclpy.parameter.PARAMETER_SEPARATOR_STRING,
                        parameter_dict=param_value,
                    )
                    parameters.extend(nested_params)
                else:
                    parameters.append(rclpy.parameter.Parameter(full_param_name, value=param_value))
            return parameters

        def set_params_from_dict(self, param_dict):
            params_to_set = self.unpack_parameter_dict("", param_dict)
            self.update(params_to_set)

        def refresh_dynamic_parameters(self):
            updated_params = self.get_params()
            # TODO remove any destroyed dynamic parameters

            # declare any new dynamic parameters

        def update(self, parameters):
            updated_params = self.get_params()

            for param in parameters:
                if param.name == self.prefix_ + "max_number_of_episodes":
                    updated_params.max_number_of_episodes = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "max_number_of_steps":
                    updated_params.max_number_of_steps = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "max_effort_command":
                    updated_params.max_effort_command = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "discount_factor":
                    updated_params.discount_factor = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "reward":
                    updated_params.reward = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

            updated_params.stamp_ = self.clock_.now()
            self.update_internal_params(updated_params)
            return SetParametersResult(successful=True)

        def update_internal_params(self, updated_params):
            self.params_ = updated_params

        def declare_params(self):
            updated_params = self.get_params()
            # declare all parameters and give default values to non-required ones
            if not self.node_.has_parameter(self.prefix_ + "max_number_of_episodes"):
                descriptor = ParameterDescriptor(description="Max number of episodes.", read_only=True)
                parameter = updated_params.max_number_of_episodes
                self.node_.declare_parameter(self.prefix_ + "max_number_of_episodes", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "max_number_of_steps"):
                descriptor = ParameterDescriptor(description="Max number of steps.", read_only=True)
                parameter = updated_params.max_number_of_steps
                self.node_.declare_parameter(self.prefix_ + "max_number_of_steps", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "max_effort_command"):
                descriptor = ParameterDescriptor(description="Effort command sent to cart.", read_only=True)
                parameter = updated_params.max_effort_command
                self.node_.declare_parameter(self.prefix_ + "max_effort_command", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "discount_factor"):
                descriptor = ParameterDescriptor(description="Discount factor.", read_only=True)
                parameter = updated_params.discount_factor
                self.node_.declare_parameter(self.prefix_ + "discount_factor", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "reward"):
                descriptor = ParameterDescriptor(description="Reward after each step.", read_only=True)
                parameter = updated_params.reward
                self.node_.declare_parameter(self.prefix_ + "reward", parameter, descriptor)

            # TODO: need validation
            # get parameters and fill struct fields
            param = self.node_.get_parameter(self.prefix_ + "max_number_of_episodes")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.max_number_of_episodes = param.value
            param = self.node_.get_parameter(self.prefix_ + "max_number_of_steps")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.max_number_of_steps = param.value
            param = self.node_.get_parameter(self.prefix_ + "max_effort_command")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.max_effort_command = param.value
            param = self.node_.get_parameter(self.prefix_ + "discount_factor")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.discount_factor = param.value
            param = self.node_.get_parameter(self.prefix_ + "reward")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.reward = param.value

            self.update_internal_params(updated_params)
