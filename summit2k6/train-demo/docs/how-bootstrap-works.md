# How the Bootstrap Builds an ACP Cluster over the Network

The bootstrap node uses the **OpenShift Agent-Based Installer** combined with **PXE boot** to provision a full ACP (OpenShift) cluster onto baremetal nodes — fully air-gapped, with all images served from a local registry.

---

## High-Level Flow

```
oc-mirror job          → populates local registry with OCP + operator images
registry-init          → sets up Harbor registry credentials
network-install job    → generates PXE files using local registry cert
tftp StatefulSet       → serves kernel/initrd to PXE-booting nodes
pubsrv Deployment      → serves rootfs over HTTP
DHCP                   → gives nodes IPs (+ points them to TFTP server)
DNS                    → resolves *.apps.bootstrap.<domain> for registry access
```

---

## Step-by-Step

### 1. MicroShift auto-starts workloads on boot

When the bootstrap node first boots, MicroShift  starts and applies all manifests under `/etc/microshift/manifests.d/`. The `network-install` manifest set kicks off the cluster provisioning process automatically.

### 2. A Kubernetes Job generates PXE boot files

The `setup-network-install` Job (`network-install/manifests/job.yaml`) runs an Ansible playbook inside an execution environment container. The playbook (`create-ocp-pxe-boot-files.yaml` stored in a ConfigMap) does the following:

1. **Pulls the registry TLS cert** from the in-cluster Harbor registry (already set up by `registry-init`)
2. **Builds the final `install-config.yaml`** — injecting the local registry auth into the pull secret and appending the registry's trust bundle as `additionalTrustBundle`
3. **Runs `openshift-install agent create pxe-files`** to generate:
   - `agent.x86_64-vmlinuz` — kernel
   - `agent.x86_64-initrd.img` — initrd with the agent installer
   - `agent.x86_64-rootfs.img` — rootfs image
4. **Copies files to two PVCs**:
   - `/tftpboot` — kernel, initrd, grub config
   - `/pubsrv` — rootfs, kubeconfig, kubeadmin-password

### 3. TFTP serves the boot files

The `tftp` StatefulSet serves files from the `/tftpboot` PVC to baremetal nodes that PXE boot. The `grub.cfg` directs nodes to the kernel and initrd.

### 4. HTTP serves the rootfs

The `pubsrv` Deployment runs an Apache httpd container that serves the rootfs image over HTTP from the `/pubsrv` PVC. An init container blocks startup until the kubeconfig file exists, ensuring the network-install Job has completed before pubsrv starts.

### 5. Target nodes PXE boot

The three target nodes (`node0`, `node1`, `arbiter`) receive static IPs and full network config from the `agent-config.yaml`. They PXE boot, pull the rootfs from pubsrv, and run the OpenShift Agent installer, which:

- Uses `rendezvousIP: NODE0_IP_ADDRESS` as the coordination point between nodes
- Installs a **2+1 topology**: 2 masters (`node0`, `node1`) + 1 arbiter (for Portworx stretch cluster quorum)
- Pulls all images from the **local bootstrap registry** via `imageContentSources` in `install-config.yaml` (no internet access required)

### 6. Post-install Jobs configure the cluster

Additional manifests baked into the `additional-manifests` ConfigMap are injected into the OpenShift install at the `openshift/` directory level and applied automatically:

| Manifest(s) | What it does |
|---|---|
| `10-catalogsource.yaml`, `11-clustercatalog.yaml` | Points OperatorHub at the local mirrored registry |
| `100-101-portworx-*.yaml` | Installs Portworx storage operator |
| `400-402-nmstate-*.yaml` | Installs NMState operator |
| `600-603-gitops-*.yaml` | Installs OpenShift GitOps (ArgoCD) |
| `04-arbiter-machineconfig.yaml` | Partitions arbiter disk (`root`, `px-metadata`) |
| `05-master-machineconfig.yaml` | Partitions master storage disk (`px-data`, `px-metadata`) |
| `06-monitoring.yaml` | Disables Alertmanager and Telemeter, sets 24h Prometheus retention |
| `9904-postinstall-nmstate` | Waits for NMState CRD, then brings up storage NICs on all three nodes via `NodeNetworkConfigurationPolicy` |
| `9999-postinstall-job` | Waits for all Cluster Operators to settle, applies `PerformanceProfile` (CPU pinning) and creates the Portworx `StorageCluster` |

---

## Cluster Topology

| Node | Role | Storage |
|---|---|---|
| `node0` | master | `px-data` + `px-metadata` partitions on dedicated storage disk |
| `node1` | master | `px-data` + `px-metadata` partitions on dedicated storage disk |
| `arbiter` | arbiter | `px-metadata` partition only (storageless KVDB tie-breaker) |

CPU partitioning is applied to all nodes via `PerformanceProfile`:
- **Reserved**: cores `0, 8`
- **Isolated**: cores `1-7, 9-15`

---

## Air-Gap Details

All OCP release images and operator catalogs are mirrored to the local Harbor registry by the `oc-mirror` Job before the network-install Job runs. The `install-config.yaml` `imageContentSources` block redirects every upstream registry reference to `registry-registry.apps.bootstrap.<BASE_DNS_ZONE>:443`, so no outbound internet access is needed during or after install.
