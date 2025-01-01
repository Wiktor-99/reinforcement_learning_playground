import genesis as gs
import os
from ament_index_python.packages import get_package_share_directory
import random

MAX_EPISODES = 100
MAX_STEPS = 500
EFFORT_CMD = 5.0
MAX_POLE_ANGLE = 1.4
MIN_POLE_ANGLE = -1.4
MAX_CART_POSITION = 0.77
MIN_CART_POSITION = -0.56


def is_simulation_truncated(cart_position, pole_angle):
    return (
        pole_angle >= MAX_POLE_ANGLE
        or pole_angle <= MIN_POLE_ANGLE
        or cart_position >= MAX_CART_POSITION
        or cart_position <= MIN_CART_POSITION
    )


def create_action(pole_angle):
    return EFFORT_CMD if pole_angle > 0 else -EFFORT_CMD


def create_scene():
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -1.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.05,
        ),
    )
    return scene


def main():
    gs.init(backend=gs.gpu, logging_level="error")
    scene = create_scene()
    scene.add_entity(gs.morphs.Plane())
    cart_pole_urdf = os.path.join(get_package_share_directory("cart_pole_bringup"), "urdf", "cart_pole.urdf")
    cart_pole = scene.add_entity(gs.morphs.URDF(file=cart_pole_urdf, fixed=True))
    scene.build()

    cart_joint_index = cart_pole.get_joint("slider_to_cart").dof_idx_local
    pole_joint_index = cart_pole.get_joint("slider_to_pole_with_holder").dof_idx_local

    for episode in range(MAX_EPISODES):
        scene.reset()
        reward = 0
        cart_pole.set_dofs_position([random.uniform(-0.05, 0.05)], [pole_joint_index])
        observation = cart_pole.get_dofs_position()
        for _ in range(MAX_STEPS):
            cart_pole.control_dofs_force([create_action(observation[pole_joint_index])], [cart_joint_index])
            scene.step()
            observation = cart_pole.get_dofs_position()
            reward += 1
            if is_simulation_truncated(observation[cart_joint_index], observation[pole_joint_index]):
                break

        print(f"Reward {reward} after {episode} episode")


if __name__ == "__main__":
    main()
