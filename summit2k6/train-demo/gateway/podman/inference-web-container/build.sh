#!/bin/bash

IMAGE_NAME=quay.io/kenosborn/inference-train-demo
IMAGE_TAG=v2

podman build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo
printf "Do you want to push this image to quay.io? (Y/N): "
read CONFIRM

case "$CONFIRM" in
  [yY]|[yY][eE][sS])
    echo "Pushing image..."
    podman push "${IMAGE_NAME}:${IMAGE_TAG}"
    ;;
  *)
    echo "Skipping push. Exiting script."
    exit 0
    ;;
esac
