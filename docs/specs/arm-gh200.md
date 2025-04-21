# Specs for `arm-gh200` BACKEND

Generated on Mon Apr 14 16:31:56 UTC 2025


💾 RAM Information
```
               total        used        free      shared  buff/cache   available
Mem:           525Gi        11Gi       497Gi        31Mi        17Gi       486Gi
Swap:             0B          0B          0B
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
Architecture:                         aarch64
CPU op-mode(s):                       64-bit
Byte Order:                           Little Endian
CPU(s):                               64
On-line CPU(s) list:                  0-63
Vendor ID:                            ARM
Model name:                           Neoverse-V2
Model:                                0
Thread(s) per core:                   1
Core(s) per cluster:                  64
Socket(s):                            -
Cluster(s):                           1
Stepping:                             r0p0
BogoMIPS:                             2000.00
Flags:                                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti
NUMA node(s):                         9
NUMA node0 CPU(s):                    0-63
NUMA node1 CPU(s):                    
NUMA node2 CPU(s):                    
NUMA node3 CPU(s):                    
NUMA node4 CPU(s):                    
NUMA node5 CPU(s):                    
NUMA node6 CPU(s):                    
NUMA node7 CPU(s):                    
NUMA node8 CPU(s):                    
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; __user pointer sanitization
Vulnerability Spectre v2:             Not affected
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected
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
overlay         3.9T   34G  3.9T   1% /
tmpfs            64M     0   64M   0% /dev
shm              64M     0   64M   0% /dev/shm
/dev/vda1       3.9T   34G  3.9T   1% /etc/hosts
tmpfs           263G  192K  263G   1% /proc/driver/nvidia
tmpfs           263G   64K  263G   1% /etc/nvidia/nvidia-application-profiles-rc.d
tmpfs            44G   28M   44G   1% /run/nvidia-persistenced/socket
tmpfs           263G     0  263G   0% /proc/acpi
tmpfs           263G     0  263G   0% /proc/scsi
tmpfs           263G     0  263G   0% /sys/firmware
```


🌐 IP Address Information
```
/root/warp-ik/scripts/specs.sh: line 24: ip: command not found
Command failed
```


🚀 NVIDIA GPU Information
```
Mon Apr 14 16:31:56 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GH200 480GB             On  |   00000000:DD:00.0 Off |                    0 |
| N/A   33C    P0             86W /  700W |       1MiB /  97871MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```


🖥️ Operating System Information
```
/root/warp-ik/scripts/specs.sh: line 24: lsb_release: command not found
Command failed
```


🐧 Kernel Information
```
Linux 193b69069337 6.8.0-1013-nvidia-64k #14lambdaguest1 SMP PREEMPT_DYNAMIC Sat Sep 14 00:46:47 UTC 2024 aarch64 aarch64 aarch64 GNU/Linux
```


🤖 NVIDIA Jetson Information
```
cat: /etc/nv_tegra_release: No such file or directory
Command failed
```


🔍 DNS Resolver Information
```
# Generated by Docker Engine.
# This file can be edited; Docker Engine will not make further changes once it
# has been modified.

nameserver 8.8.8.8
search ll.local

# Based on host file: '/run/systemd/resolve/resolv.conf' (legacy)
# Overrides: []
```

Number of GPUs: 1
DEVICE NVIDIA GH200 480GB (id=0)
Device compute capability: major=9, minor=0
Architecture: sm_90 (Hopper)
PCI bus id=0000:DD:00.0
Properties:
------------
aarch64
Properties:
------------
- Can map host memory into the CUDA address space: YES
- Can access host registered memory at the same virtual address as the CPU: YES
- Clock rate: 1.98GHz
- Peak memory clock frequency: 2.62GHz
- Performance ratio (single precision)/(double precision): 2
- Compute capability: major=9, minor=0
- Compute mode: 0 (0 - default, 2 - prohibited, 3 - exclusive process)
- Support for Compute Preemption: YES
- Support for concurrent kernels execution within the same context: YES
- Support for coherent access to managed memory concurrently with CPU: YES
- Support for deferred mapping in CUDA arrays and CUDA mipmapped arrays: YES
- Support for direct access of managed memory on device without migration: YES
- ECC enabled: YES
- Support for generic compression: YES
- Support for caching globals in L1 cache: YES
- Support for caching locals in L1 cache: YES
- Global memory bus widths: 6144 bits
- Support for GPUDirect RDMA: YES
- GPUDirect RDMA flush-writes options bitmask: 0b00000000000000000000000000000011
- GPUDirect RDMA writes ordering: 200 (0 - none, 100 - this device can consume remote writes, 200 - any CUDA device can consume remote writes to this device)
- Can concurrently copy memory between host and device while executing kernel: YES
- Support for exporting memory to a posix file descriptor: YES
- Support for exporting memory to a Win32 NT handle: NO
- Support for exporting memory to a Win32 KMT handle: NO
- Link between device and host supports native atomic operations: YES
- Device is integrated with memory subsystem: NO
- Kernel execution timeout: NO
- L2 cache size: 60.00MB
- Max L2 persisting lines capacity: 37.50MB
- Support for managed memory allocation: YES
- Max access policy window size: 128.00MB
- Max x-dimension of a block: 1024
- Max y-dimension of a block: 1024
- Max z-dimension of a block: 64
- Max blocks in a multiprocessor: 32
- Max x-dimension of a grid: 2147483647
- Max y-dimension of a grid: 65535
- Max z-dimension of a grid: 65535
- Max pitch allowed by the memory copy functions: 2.00GB
- Max number of 32-bit registers per block: 65536
- Max number of 32-bit registers in a multiprocessor: 65536
- Max shared memory per block: 49152B
- Max optin shared memory per block: 232448B
- Max shared memory available to a multiprocessor: 233472B
- Max threads per block: 1024
- Max threads per multiprocessor: 2048
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
- Bitmask of handle types supported with memory pool-based IPC: 0b00000000000000000000000000001001
- Multi-GPU board: NO
- Multi-GPU board group ID: 0
- Support for switch multicast and reduction operations: NO
- Number of multiprocessors: 132
- NUMA configuration: 1
- NUMA node ID of GPU memory: 1
- Support for coherently accessing pageable memory: YES
- Access pageable memory via host's page tables: YES
- PCI bus ID: 221
- PCI device (slot) ID: 0
- PCI domain ID: 0
- Support for registering memory that must be mapped to GPU as read-only: NO
- Amount of shared memory per block reserved by CUDA driver: 1024B
- Support for sparse CUDA arrays and sparse CUDA mipmapped arrays: YES
- Using TCC driver: NO
- Constant memory available: 65536B
- Support for unified address space with host: YES
- Support for virtual memory management: YES
*****************************************************