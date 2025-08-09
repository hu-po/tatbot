# Tatbot Network Refactor: Balanced Secure and Convenient Design

## Context and Goals

This design combines the strengths of both proposed refactor plans to create a clean, baseline implementation for the tatbot prototype robotic system. It's not mission-critical, so we prioritize simplicity, reliability, and basic security over enterprise-grade features like full mTLS or complex VLANs. From Plan 1, we adopt automated service discovery elements (simplified), device abstraction ideas, and health monitoring. From Plan 2, we take centralized DNS/DHCP management, inventory via nodes.yaml, fast idempotent toggles, and practical hardening like limited key distribution and firewalls.

- **Heterogeneous Devices**: RPis (e.g., rpi1, rpi2), Jetsons/PCs (e.g., ook, ojo), Trossen arm controllers (e.g., trossen-ai, arm-l/r), IP cameras (camera1-5). Linux devices support SSH; others use vendor UIs/APIs.
- **Operating Modes**:
  - **Edge Mode**: Self-contained isolated network for standalone operation. All devices resolve and reach each other (e.g., MCP servers) without external connectivity.
  - **Home Mode**: Integrated with home network (192.168.1.0/24) for developer access from oop (e.g., 192.168.1.51). Devices use home router for DHCP/DNS; minimal special infra.
- **Goals**: Single-command mode toggle (fast, idempotent, with rollbacks); basic security (e.g., no widespread shared keys, restricted access); automatic service discovery; easy setup/modification. Cover basics without overkill.

## Current Situation Summary

Drawing from both plans:
- Shared SSH keys enable lateral movement risk.
- Mode switching is manual/slow (per-node SSH, no atomicity/rollback).
- Hardcoded IPs/ports hinder discovery; heterogeneous configs complicate management.
- Open access exposes cameras/arms; no segmentation or firewalls.
- Drift in tools/docs (e.g., DNS on rpi1 vs. rpi2).

## Guiding Principles

- **Centralize Changes**: Mode toggles affect primarily one anchor node (DNS/DHCP), minimizing per-device touches.
- **Single Source of Truth**: Use `nodes.yaml` for inventory (MACs, IPs, hostnames, roles, capabilities).
- **Balanced Security**: Limit secrets (management SSH key only, no private key copying); enable basic firewalls; change default credentials. Skip advanced mTLS/VLANs for prototype simplicity.
- **Operational Simplicity**: Single CLI command for toggles/checks; automatic discovery via DNS; health verification with rollbacks.
- **Fail-Safe**: Preflight validation, post-toggle checks, easy rollback.
- **Modularity**: Build on existing scripts (e.g., tatbot.utils.net/mode_toggle); extend for new features.

## Optimal Network Topology

We use a hybrid approach: Plan 2's flat subnet for simplicity, with Plan 1's isolated subnet in edge mode and bridged integration in home mode. Anchor node is rpi1 (static IP: 192.168.1.53 in home, 192.168.100.1 in edge) for DNS/DHCP/CA-lite (if needed for future extensions).

### Edge Mode: Isolated Self-Contained Network
Uses a dedicated subnet (192.168.100.0/24) to avoid home network conflicts. All devices connect via a tatbot switch/router; no external access by default.

```
┌─────────────────────────────────────────────────────────────┐
│ Edge Mode (192.168.100.0/24) - Isolated                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  rpi1 (100.1) ─┐                                          │
│  DNS/DHCP      │ ┌─ trossen-ai (100.10) ─ arm-l (100.11) │
│  Anchor/Node    │ │               └─ arm-r (100.12)      │
│                 ├─┤                                        │
│  rpi2 (100.2) ──┘ ├─ ojo (100.20) ─ realsense1/2         │
│  NFS/Backup       │                                        │
│                    ├─ ook (100.30)                         │
│                    │                                        │
│                    └─ camera1-5 (100.40-44)               │
│                                                             │
│ Communication: DNS resolves *.tatbot.local; MCP via A/SRV  │
│ records; firewalls restrict to subnet.                     │
└─────────────────────────────────────────────────────────────┘
```

- **Behavior**: rpi1 serves DHCP (fixed leases from nodes.yaml) and DNS (authoritative for tatbot.local). NFS on rpi2 for shared storage. All nodes mount by hostname (not IP). Optional upstream DNS if WAN plugged in.

### Home Mode: Integrated with Home Network
Devices join the home network (192.168.1.0/24) via the home router. Use a simple bridge (e.g., tatbot-bridge device or software on rpi1) for tatbot devices if segmentation desired; otherwise, flat integration.

```
┌───────────────────────────────────────────────────────────────┐
│ Home Network (192.168.1.0/24)                                │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Home Router (1.1) ─┬─ oop (1.51) ← Developer Access        │
│                     │                                         │
│                     ├─ tatbot-bridge (1.100, optional)      │
│                     │   ↓ Secure Access (firewall rules)    │
│                     │                                         │
│  ┌──────────────────┼─────────────────────────────────────┐   │
│  │ Tatbot Devices (dynamic/static IPs from router)        │   │
│  │                  │                                    │   │
│  │  rpi1 (1.53) ────┤  ook (dynamic)                     │   │
│  │  rpi2 (dynamic) ─┤  ojo (dynamic)                     │   │
│  │  trossen-ai ─────┤  cameras (dynamic/static)          │   │
│  │  (dynamic)       │  arms (dynamic)                    │   │
│  └──────────────────┼─────────────────────────────────────┘   │
│                     │                                         │
└───────────────────────────────────────────────────────────────┘
```

- **Behavior**: Home router handles DHCP/DNS; rpi1 runs dnsmasq as forwarder for local tatbot.local resolutions. oop reaches all via hostnames. Internet access for updates. Firewalls limit inbound to trusted IPs (e.g., oop).

## Key Design Decisions

1. **Anchor Node (rpi1)**: Centralizes DNS/DHCP via dnsmasq with mode profiles (home: forwarder; edge: authoritative). Static IP in both modes for reliability. Chosen over rpi2 to match some docs; easy to change.
   
2. **IP and Inventory Management**: Flat subnet per mode; static reservations in nodes.yaml (MAC, IP, hostname, role, capabilities like "dns_server" or "robot_control"). Generate dnsmasq configs from this file. Hostnames consistent (*.tatbot.local) across modes.

3. **Service Discovery**: Basic DNS A records for nodes; add SRV records for MCP services (e.g., _mcp._tcp.tatbot.local). Skip full Consul for prototype; use a simple services.yaml if needed for ports.

4. **Security Basics**:
   - **SSH**: Generate management keypair on operator host (e.g., ook); append public key to authorized_keys on nodes. No private key distribution; purpose-limited keys for inter-node if required.
   - **Firewalls**: Enable UFW/nftables on Linux nodes; allow SSH/MCP from tatbot subnet only (edge) or trusted IPs (home). Change camera defaults; disable unnecessary remote access.
   - **Credentials**: Avoid hardcoding; use .env with gitignore. No auto-rotation yet.
   - Skip mTLS/VLANs: Too heavy for prototype; add later if needed.

5. **Mode Switching**: Atomic via symlink swap on anchor (dnsmasq reload). CLI: `tatctl mode set edge --verify`. Idempotent; rollback on failure.

6. **Automation and CLI**: Extend existing tatbot scripts into `tatctl` for mode set/status/check, mcp restart, net render. Include health checks (DNS resolve, port open, MCP alive).

7. **Heterogeneous Handling**: Device profiles in nodes.yaml (e.g., config_methods: dhcpcd/netplan). Abstract toggles: For non-SSH devices (cameras), set DNS once manually.

## Implementation TODO List

- Document anchor (rpi1, IP 192.168.1.53/192.168.100.1).
- Collect MACs/IPs/roles; create `nodes.yaml`.
- Install dnsmasq on rpi1; create mode-home.conf (forwarder to home router) and mode-edge.conf (DHCP range 192.168.100.100-200, static leases, A/SRV records).
- Implement `tatctl net render` to generate configs from nodes.yaml (with dnsmasq --test validation).
- Setup symlink for active.conf; create /usr/local/sbin/tatbot-mode on rpi1 for swap/reload.
- Harden SSH: Generate management key; distribute public only via script.
- Enable firewalls on Linux nodes (allow tatbot subnet/home trusted).
- Change camera passwords/disable extras.
- Add SRV records for MCP; implement simple discovery in clients (query DNS).
- Update NFS/clients to use hostnames.
- Extend CLI: `tatctl mode set|status|check`, `tatctl mcp restart --all`.
- Add preflight (SSH to anchor, config test); postflight (resolve hostnames, check ports/MCP health).
- Implement rollback: Backup symlink before swap; revert on failure.
- Test toggles: Verify no downtime for internal services.
- Add logs/observability (journald for dnsmasq; `tatctl logs`).
- Document processes below.
- Optional: Basic health dashboard on rpi1.

## Process to Setup, Verify, and Modify

### Setup Process
1. **Inventory**: Edit `nodes.yaml` with device details.
2. **Generate Configs**: Run `tatctl net render --dry-run` locally; then deploy to rpi1.
3. **Install/Configure Anchor**: SSH to rpi1; install dnsmasq; copy configs/symlink.
4. **Device Prep**: Set DNS on static devices (cameras/arms) to anchor IP. Enable DHCP on others.
5. **Security**: Run SSH hardening script; enable firewalls; update credentials.
6. **Initial Toggle**: `tatctl mode set home --verify` (default).

### Verify Process
- **Status Check**: `tatctl mode status` (queries DNS, lists mode/resolutions).
- **Health Check**: `tatctl check --all` (parallel: DNS resolve for all nodes, MCP ports open, service pings).
- **Post-Toggle**: Automatic in `set` command: Wait 10-30s, run checks; rollback if fails (e.g., <50% nodes resolve).
- **Manual**: SSH to nodes; use `dig/nslookup` for DNS, `nc/telnet` for ports.

### Modify Process
1. **Add/Change Device**: Update `nodes.yaml` (e.g., new camera MAC/IP/hostname).
2. **Re-Render**: `tatctl net render`; deploy to rpi1.
3. **Reload**: If in edge, reload dnsmasq; devices pick up via lease renewal.
4. **Mode-Agnostic**: Changes apply consistently; test with `tatctl check`.
5. **Extend Features**: Add to nodes.yaml (e.g., new capability); update CLI/scripts as needed.

## Risks and Mitigations
- **Anchor Failure**: Mitigate with backup config on rpi2; manual failover.
- **DNS Caching**: Short TTLs (1m during transition); force reloads on devices.
- **Toggle Downtime**: Limit to seconds; internal services use hostnames for resilience.
- **Security Gaps**: Basics covered; audit logs optional for prototype.

## Benefits
- **Security**: Reduced risks (limited keys, firewalls); better than current without complexity.
- **Convenience**: Fast toggles (seconds), auto-discovery, single CLI.
- **Prototype Fit**: Clean, extensible; covers basics for robotics research.

This design delivers a balanced refactor: Secure enough for prototype, convenient for daily use, and easy to evolve.