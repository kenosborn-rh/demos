# Build Podman Container
podman build -f Containerfile -t placard-inference:v4 .

# Build Podman Conatiner (no-cache)
podman build --no-cache -f Containerfile -t placard-inference:v4 .

# Check python versioning
# podman run --rm placard-inference:v4 python3 --version

# Tag for Quay registry (thinking we use OCP registry instead for stuff)
podman tag localhost/placard-inference:v4 quay.io/kenosborn/placard-inference:v4

# Login to Quay.io
podman login quay.io

# Push
podman push quay.io/kenosborn/placard-inference:v4

