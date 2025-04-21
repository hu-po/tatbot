# Specs for `arm-agx` BACKEND

Generated on Mon Apr 14 16:36:59 UTC 2025


💾 RAM Information
```
               total        used        free      shared  buff/cache   available
Mem:            29Gi       1.3Gi        13Gi        29Mi        14Gi        28Gi
Swap:           15Gi          0B        15Gi
```


📡 Routing Information
```
/root/warp-ik/scripts/specs.sh: line 24: ip: command not found
Command failed
```


🎮 GPU Information
```
/root/warp-ik/scripts/specs.sh: line 24: lspci: command not found
Command failed
```


🧠 CPU Information
```
Architecture:                       aarch64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
CPU(s):                             12
On-line CPU(s) list:                0-11
Vendor ID:                          ARM
Model name:                         Cortex-A78AE
Model:                              1
Thread(s) per core:                 1
Core(s) per cluster:                4
Socket(s):                          -
Cluster(s):                         3
Stepping:                           r0p1
CPU max MHz:                        2201.6001
CPU min MHz:                        115.2000
BogoMIPS:                           62.50
Flags:                              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp uscat ilrcpc flagm paca pacg
L1d cache:                          768 KiB (12 instances)
L1i cache:                          768 KiB (12 instances)
L2 cache:                           3 MiB (12 instances)
L3 cache:                           6 MiB (3 instances)
NUMA node(s):                       1
NUMA node0 CPU(s):                  0-11
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Not affected
Vulnerability Spec rstack overflow: Not affected
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:           Mitigation; __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; CSV2, but not BHB
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected
```


🐋 Docker Version
```
/root/warp-ik/scripts/specs.sh: line 24: docker: command not found
Command failed
```


🐳 Docker Information
```
/root/warp-ik/scripts/specs.sh: line 24: docker: command not found
Command failed
```


💽 Disk Information
```
Filesystem      Size  Used Avail Use% Mounted on
overlay         916G   55G  815G   7% /
tmpfs            64M     0   64M   0% /dev
shm              64M     0   64M   0% /dev/shm
/dev/nvme0n1    916G   55G  815G   7% /etc/hosts
/dev/mmcblk0p1   57G   27G   27G  50% /usr/sbin/nvidia-smi
tmpfs            15G     0   15G   0% /proc/asound
tmpfs            15G     0   15G   0% /sys/firmware
```


🌐 IP Address Information
```
/root/warp-ik/scripts/specs.sh: line 24: ip: command not found
Command failed
```


🚀 NVIDIA GPU Information
```
Mon Apr 14 16:36:59 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 540.4.0                Driver Version: 540.4.0      CUDA Version: 12.6     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Orin (nvgpu)                  N/A  | N/A              N/A |                  N/A |
| N/A   N/A  N/A               N/A /  N/A | Not Supported        |     N/A          N/A |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```


🖥️ Operating System Information
```
/root/warp-ik/scripts/specs.sh: line 24: lsb_release: command not found
Command failed
```


🐧 Kernel Information
```
Linux bee79cf9812d 5.15.148-tegra #1 SMP PREEMPT Tue Jan 7 17:14:38 PST 2025 aarch64 aarch64 aarch64 GNU/Linux
```


🤖 NVIDIA Jetson Information
```
# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```


🔍 DNS Resolver Information
```
# Generated by Docker Engine.
# This file can be edited; Docker Engine will not make further changes once it
# has been modified.

nameserver 192.168.1.1
nameserver 192.168.1.1
search .

# Based on host file: '/run/systemd/resolve/resolv.conf' (legacy)
# Overrides: []
```

Number of GPUs: 1
DEVICE Orin (id=0)
Device compute capability: major=8, minor=7
Architecture: sm_87 (Ampere)
PCI bus id=0000:00:00.0
Properties:
------------
aarch64
Properties:
------------
- Can map host memory into the CUDA address space: YES
- Can access host registered memory at the same virtual address as the CPU: NO
- Clock rate: 1.30GHz
- Peak memory clock frequency: 0.82GHz
- Performance ratio (single precision)/(double precision): 32
- Compute capability: major=8, minor=7
- Compute mode: 0 (0 - default, 2 - prohibited, 3 - exclusive process)
- Support for Compute Preemption: YES
- Support for concurrent kernels execution within the same context: YES
- Support for coherent access to managed memory concurrently with CPU: NO
- Support for deferred mapping in CUDA arrays and CUDA mipmapped arrays: NO
- Support for direct access of managed memory on device without migration: NO
- ECC enabled: NO
- Support for generic compression: YES
- Support for caching globals in L1 cache: YES
- Support for caching locals in L1 cache: YES
- Global memory bus widths: 256 bits
- Support for GPUDirect RDMA: YES
- GPUDirect RDMA flush-writes options bitmask: 0b00000000000000000000000000000001
- GPUDirect RDMA writes ordering: 200 (0 - none, 100 - this device can consume remote writes, 200 - any CUDA device can consume remote writes to this device)
- Can concurrently copy memory between host and device while executing kernel: YES
- Support for exporting memory to a posix file descriptor: YES
- Support for exporting memory to a Win32 NT handle: NO
- Support for exporting memory to a Win32 KMT handle: NO
- Link between device and host supports native atomic operations: YES
- Device is integrated with memory subsystem: YES
- Kernel execution timeout: NO
- L2 cache size: 4.00MB
- Max L2 persisting lines capacity: 2.75MB
- Support for managed memory allocation: YES
- Max access policy window size: 128.00MB
- Max x-dimension of a block: 1024
- Max y-dimension of a block: 1024
- Max z-dimension of a block: 64
- Max blocks in a multiprocessor: 16
- Max x-dimension of a grid: 2147483647
- Max y-dimension of a grid: 65535
- Max z-dimension of a grid: 65535
- Max pitch allowed by the memory copy functions: 2.00GB
- Max number of 32-bit registers per block: 65536
- Max number of 32-bit registers in a multiprocessor: 65536
- Max shared memory per block: 49152B
- Max optin shared memory per block: 166912B
- Max shared memory available to a multiprocessor: 167936B
- Max threads per block: 1024
- Max threads per multiprocessor: 1536
- Warp size: 32
- Max 1D surface width: 32768
- Max layers in 1D layered surface: 2048
- Max 1D layered surface width: 32768
- Max 2D surface width: 131072
- Max 2D surface height: 65536
- Max layers in 2D layered surface: 2048
- Max 2D layered surface width: 32768
- Max 2D layered surface height: 32768
- Max 3D surface width: 16384
- Max 3D surface height: 16384
- Max 3D surface depth: 16384
- Max cubemap surface width: 32768
- Max layers in a cubemap layered surface: 2046
- Max cubemap layered surface width: 32768
- Max 1D texture width: 131072
- Max width for a 1D texture bound to linear memory: 268435456
- Max layers in 1D layered texture: 2048
- Max 1D layered texture width: 32768
- Max mipmapped 1D texture width: 32768
- Max 2D texture width: 131072
- Max 2D texture height: 65536
- Max width for a 2D texture bound to linear memory: 131072
- Max height for a 2D texture bound to linear memory: 65000
- Max pitch for a 2D texture bound to linear memory: 2.00MB
- Max layers in 2D layered texture: 2048
- Max 2D layered texture width: 32768
- Max 2D layered texture height: 32768
- Max mipmapped 2D texture width: 32768
- Max mipmapped 2D texture height: 32768
- Max 3D texture width: 16384
- Max 3D texture height: 16384
- Max 3D texture depth: 16384
- Alternate max 3D texture width: 8192
- Alternate max 3D texture height: 8192
- Alternate max 3D texture depth: 32768
- Max cubemap texture width or height: 32768
- Max layers in a cubemap layered texture: 2046
- Max cubemap layered texture width or height: 32768
- Texture base address alignment requirement: 512B
- Pitch alignment requirement for 2D texture references bound to pitched memory: 32B
- Support for memory pools: YES
- Bitmask of handle types supported with memory pool-based IPC: 0b00000000000000000000000000000000
- Multi-GPU board: NO
- Multi-GPU board group ID: 0
- Support for switch multicast and reduction operations: NO
- Number of multiprocessors: 16
- NUMA configuration: None
- NUMA node ID of GPU memory: None
- Support for coherently accessing pageable memory: NO
- Access pageable memory via host's page tables: NO
- PCI bus ID: 0
- PCI device (slot) ID: 0
- PCI domain ID: 0
- Support for registering memory that must be mapped to GPU as read-only: YES
- Amount of shared memory per block reserved by CUDA driver: 1024B
- Support for sparse CUDA arrays and sparse CUDA mipmapped arrays: NO
- Using TCC driver: NO
- Constant memory available: 65536B
- Support for unified address space with host: YES
- Support for virtual memory management: YES
*****************************************************