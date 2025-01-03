FROM althack/ros2:jazzy-full
ARG ROS_DISTRO=jazzy
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV SHELL /bin/bash
ENV DEBIAN_FRONTEND=noninteractive
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y wget bash-completion git-core pre-commit

RUN sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update --fix-missing && apt-get -y install \
    ros-$ROS_DISTRO-ros-gz \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
    ros-$ROS_DISTRO-generate-parameter-library-py \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install tensorflow control sympy scipy ipykernel nbconvert==7.16.1 nbformat --break-system-packages torch
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git && \
    cd Genesis && pip install --break-system-packages -e .
ENV DEBIAN_FRONTEND=dialog

ARG WORKSPACE
RUN echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc \
    && echo "if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash; fi" >> /home/$USERNAME/.bashrc \
    && echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> //home/$USERNAME/.bashrc
