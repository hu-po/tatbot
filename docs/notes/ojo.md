NVIDIA Jetson AGX Orin Developer Kit Processor: NVIDIA Ampere (2048 CUDA cores, 64 Tensor cores), 12-core ARM Cortex-A78AE v8.2 CPU AI Performance: 2× NVDLA 2.0; PVA 2.0 Memory & Storage: 32GB 256-bit LPDDR5; 64GB eMMC 5.1 Power: Includes AC adapter with USB-C cord, plus USB-C to USB-A cord I/O: 16-lane MIPI CSI-2 camera connector x16 PCIe (supporting x8 PCIe Gen4) USB: 2× Type-C (USB 3.2 Gen2, PD), 3× Type-A (USB 3.2 Gen2) microSD slot Networking: 10 GbE RJ45 M.2 Slots: Key M (NVMe), Key E (Wi-Fi) Display: DisplayPort 1.4a (MST support) Headers & Connectors: 40-pin (GPIO, SPI, CAN, I2S, UART, DMIC) 12-pin automation header 10-pin audio header 10-pin JTAG header 4-pin fan header 2-pin RTC battery connector Buttons: Power, Force Recovery, Reset Model Info: Model P3701; SN 1427220530466; PN 951-13700-000-000

how to know specific version:
https://forums.developer.nvidia.com/t/jetson-agx-orin-64-dev-kit-only-32gb-available/296679/10?u=hu.po.xyz


```bash
sudo nvpmodel -q --verbose # check current mode
sudo nvpmodel -m 0 # max power mode
```

available power modes

| Power Mode Name | ID  | Description                                                                 |
|-----------------|-----|-----------------------------------------------------------------------------|
| MAXN            | 0   | Maximum performance mode with no constraints, allowing the system to use its full potential. |
| MODE_15W        | 1   | Low-power mode limiting the system to a 15W power envelope, ideal for energy-efficient operations. |
| MODE_30W        | 2   | Balanced power mode with a 30W power envelope, offering a good balance between performance and power usage. |
| MODE_50W        | 3   | High-performance mode allowing the system to operate within a 50W power envelope for increased performance. |

```bash
ojo@ubuntu:~$ cat /etc/nv_boot_control.conf
TNSPEC 3701-500-0000-M.0-1-1-jetson-agx-orin-devkit-
COMPATIBLE_SPEC 3701-300-0000--1--jetson-agx-orin-devkit-
TEGRA_BOOT_STORAGE mmcblk0
TEGRA_CHIPID 0x23
TEGRA_OTA_BOOT_DEVICE /dev/mtdblock0
TEGRA_OTA_GPT_DEVICE /dev/mtdblock0
```