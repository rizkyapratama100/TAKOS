#!/bin/bash

# Configuration
IMAGE_NAME="takos-ros2"
CONTAINER_NAME="takos_dev"
WORKSPACE_DIR="/workspaces/TAKOS"

# 1. Provide X11 permissions for the GUI
# Run this on the host to allow Docker to connect to the display
if command -v xhost > /dev/null; then
    xhost +local:docker > /dev/null
fi

# 2. Build the image if it doesn't exist
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .devcontainer/
fi

# 3. Check if container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container is already running. Opening a new shell..."
    docker exec -it $CONTAINER_NAME bash
    exit 0
fi

# 4. Start the container
echo "Starting $CONTAINER_NAME..."

docker run -it \
    --name $CONTAINER_NAME \
    --rm \
    --privileged \
    --net=host \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v "$(pwd):$WORKSPACE_DIR" \
    -w "$WORKSPACE_DIR" \
    $IMAGE_NAME \
    bash -c "source /opt/ros/jazzy/setup.bash && git config --global --add safe.directory $WORKSPACE_DIR && exec bash"
