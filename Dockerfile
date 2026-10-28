# Official ROS 2 Jazzy runtime; CPU compatible. ROS-specific integration remains a later stage.
FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv ros-jazzy-rtabmap-ros \
    ros-jazzy-rosbag2 ros-jazzy-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml /workspace/
COPY src /workspace/src
RUN python3 -m venv --system-site-packages /opt/semantic-venv \
    && /opt/semantic-venv/bin/pip install --no-cache-dir -e '.[rgbd]'
ENV PATH="/opt/semantic-venv/bin:$PATH"
CMD ["bash"]
