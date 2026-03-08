```
podman run --rm -it \
  --privileged \
  --pull=newer \
  --security-opt label=type:unconfined_t \
  -v ./config.toml:/config.toml:ro \
  -v ./output:/output \
  -v /var/lib/containers/storage:/var/lib/containers/storage \
  quay.io/centos-bootc/bootc-image-builder:latest \
  build \
  --type anaconda-iso \
  --output /output \
  quay.io/ken/gateway-summit2k6-bootc:v1
```
