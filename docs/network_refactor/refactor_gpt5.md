## Tatbot network refactor: secure and convenient modes (home vs edge)

### Context and goals

- **Heterogeneous devices**: RPis, Jetson/PCs, Trossen arm controllers, IP cameras. Some run Linux with SSH, others have vendor UIs only. Services (e.g., MCP servers) run across nodes.
- **Two operating modes**:
  - **Edge mode**: All arms, IP cameras, and tatbot nodes form a self-contained network. They must resolve each other, reach each other’s MCP servers, and continue operating even without the home network.
  - **Home mode**: All devices use the home router (DHCP + DNS). Everything should be reachable from the home desktop `oop`. Minimal special infra.
- **Goals**: One-command toggle, fast, idempotent, safe rollbacks, and secure defaults. No per-device manual edits during toggles.

### Current situation (observations)

- SSH orchestration exists via `tatbot.utils.net` to generate and distribute a shared key and build `~/.ssh/config`. The current flow uploads both public and private keys to remote nodes for frictionless inter-node SSH, which is convenient but increases lateral movement risk.
- DNS toggle script (`tatbot.utils.mode_toggle`) connects to each node over SSH to edit `dhcpcd.conf` and start/stop `dnsmasq` on a chosen DNS node. This is brittle (device variety), slow (per-node SSH), and harder to validate/rollback.
- Docs indicate in edge mode the DNS server is `rpi1`, while the toggler defaults to `rpi2`. There is potential drift between tooling and documentation.
- NFS is hosted on `rpi2` with clients mounting by IP. That works but couples mounts to a single IP and requires correct DNS in edge mode for convenience.

### Security vs convenience: guiding principles

- **Centralize state changes**: Toggling should modify a single, well-understood component (the DNS/DHCP node) rather than touching every device.
- **Single source of truth**: Maintain host inventory and static IP/MAC/role metadata in `nodes.yaml` and generate network configs from it.
- **Limit secrets replication**: Prefer appending public keys to `authorized_keys` over copying private keys to all nodes. If inter-node SSH is needed, scope it tightly (per-role keys).
- **Defense in depth**: Enforce basic host firewalls (UFW/nftables) to restrict inbound services to the tatbot LAN in edge mode and to trusted subnets in home mode.
- **Fail fast and rollback**: Validate DNS/DHCP config before reload; perform postflight checks; roll back if validation fails.

### Target design

1) DNS/DHCP anchor node with stable IP

- Pick a dedicated “network control node” (recommended: `rpi1`) with a static IP (e.g., 192.168.1.53) on the tatbot LAN.
- Install `dnsmasq` and manage two profiles:
  - `mode-home.conf`: DNS forwarder pointing to the home router (or public DNS), DHCP disabled.
  - `mode-edge.conf`: Authoritative for `tatbot.local`, DHCP enabled for the tatbot LAN, fixed leases for known MACs, and upstream DNS for internet if available.
- Profile selection via symlink: `/etc/dnsmasq.d/active.conf -> mode-home.conf | mode-edge.conf`.
- Toggle by swapping the symlink and reloading `dnsmasq`. Validate config with `dnsmasq --test` before reload.

2) IP plan and inventory

- Define a flat subnet (e.g., 192.168.1.0/24) with static reservations for critical nodes:
  - Core nodes: `ook`, `ojo`, `trossen-ai`, `rpi1`, `rpi2`, arms (`arm-l`, `arm-r`), cameras (`camera1..5`).
- Store MAC addresses and desired IPs/hostnames in `nodes.yaml`. Generate `dnsmasq` static DHCP mappings and `address=` records from this file.
- Keep hostnames consistent across modes, e.g., `ook.tatbot.local`, `arm-l.tatbot.local`.

3) Mode semantics

- Edge mode (isolated, self-contained):
  - `dnsmasq` on `rpi1` serves DHCP for the tatbot switch fabric and DNS authoritative for `tatbot.local`.
  - All devices receive Option 6 (DNS) = 192.168.1.53 and fixed IPs when possible. Devices with static config (some IP cameras/arm boxes) are set once to use 192.168.1.53 for DNS.
  - Optional upstream DNS (e.g., `server=1.1.1.1`) if WAN is present; otherwise, the system functions without internet.
  - All MCP servers are discoverable by DNS and reachable via static ports.

- Home mode (integrated into home LAN):
  - `dnsmasq` is switched to the home profile (no DHCP) or fully stopped; the home router provides DHCP/DNS.
  - Devices get IPs from the home router and are reachable from `oop`.
  - To preserve convenience, maintain the same hostnames by either:
    - letting the router serve local DNS (if supported), or
    - keeping `dnsmasq` as a pure DNS forwarder on 192.168.1.53 and setting DHCP Option 6 on the router to point clients to 192.168.1.53.
  - If the second option is used, toggling remains centralized while still “using the home router” for addressing.

4) MCP service discovery

- Standardize MCP ports per role (e.g., arms on 5173, vision on 5180, coordinator on 5190), or publish DNS-SD SRV records in `dnsmasq` if clients support it.
- At minimum, include `A` records for all nodes and a simple static `services.yaml` mapping hostname -> MCP port per service.
- Provide a `tatctl mcp restart --all` command to use `ssh <node> "bash scripts/run_mcp.sh <node>"` for farm-wide restarts.

5) Security hardening

- SSH:
  - Stop distributing the private key by default. Instead, generate a management keypair on the operator host (`ook`) and append its public key to `authorized_keys` on nodes.
  - If inter-node SSH is required (e.g., arms contacting controllers), generate purpose-limited keys and distribute only those public keys to specific targets.
  - Enforce `~/.ssh/config` with `IdentitiesOnly yes` and limited `Host` stanzas.
- Firewalls:
  - Enable UFW/nftables on Linux hosts. Allow inbound SSH and MCP ports from the tatbot subnet only. Deny by default.
  - On cameras with vendor controls, disable remote access features and change default passwords.
- Services:
  - Run MCP servers under non-root users, with systemd units and hardened sandboxing where possible.
  - Keep `dnsmasq` confined to LAN interfaces.

6) Operations and tooling

- CLI: `tatctl` (or extend existing scripts) with subcommands:
  - `tatctl mode set home|edge` → SSH into `rpi1`, run `sudo tatbot-mode <mode>`.
  - `tatctl mode status` → Resolve a set of hostnames and verify port 53 on `rpi1`.
  - `tatctl check` → Parallel health checks: DHCP lease present (edge), DNS resolves, MCP ports open per node.
  - `tatctl mcp restart --all|<role>|<node>` → Farm-wide restart via the provided `scripts/run_mcp.sh`.
- DNS/DHCP config generation:
  - `tatctl net render` → Read `nodes.yaml` and write `/etc/dnsmasq.d/mode-edge.conf` with `dhcp-host=` lines (MAC, IP, hostname) and `address=` records.
  - Include preflight validation (`dnsmasq --test`) and dry-run.
- Logs and observability:
  - Forward dnsmasq logs to journald; add a simple `tatctl mode logs` tail.
  - Export a minimal status page on `rpi1` (optional) showing leases and DNS hits.

7) Rollback and validation

- Preflight: ensure SSH to `rpi1` works, `dnsmasq` installed, configs render cleanly.
- Toggle: update symlink, validate, reload; if reload fails, revert symlink.
- Postflight: parallel resolution checks (e.g., `ook.tatbot.local`, `arm-l.tatbot.local`, cameras), verify DHCP lease (edge), confirm MCP ports are reachable.

### Example dnsmasq snippets

Home profile (no DHCP; pure forwarder):

```
no-resolv
server=192.168.1.1
localise-queries
domain=tatbot.local
```

Edge profile (authoritative DHCP + DNS):

```
interface=eth0
bind-interfaces
domain=tatbot.local
expand-hosts
local=/tatbot.local/
no-resolv
server=1.1.1.1

# DHCP for 192.168.1.0/24 with static reservations generated from nodes.yaml
dhcp-range=192.168.1.100,192.168.1.200,12h
# Examples (generated):
# dhcp-host=AA:BB:CC:DD:EE:FF,ook,192.168.1.10
# dhcp-host=11:22:33:44:55:66,trossen-ai,192.168.1.20

# A records (generated):
# address=/ook.tatbot.local/192.168.1.10
# address=/arm-l.tatbot.local/192.168.1.30
```

### Migration plan

1) Decide and document the DNS/DHCP anchor node (`rpi1`) and its static IP (e.g., 192.168.1.53).
2) Collect MAC addresses and intended IPs for all devices; update `nodes.yaml`.
3) Implement `tatctl net render` to generate `mode-edge.conf` from `nodes.yaml`.
4) Install `dnsmasq` on `rpi1` and set up `mode-home.conf`, `mode-edge.conf`, and the `active.conf` symlink.
5) Create `/usr/local/sbin/tatbot-mode` on `rpi1` to swap modes with validation and reload.
6) Switch devices to obtain DNS from either the router (home) or `rpi1` (edge). For static devices, set DNS once to 192.168.1.53.
7) Implement `tatctl mode set|status|check` and `tatctl mcp restart` wrappers using the existing `scripts/` and `uv` environment.
8) Harden SSH and firewalls, rotate passwords on cameras, and ensure least privilege for services.

### Risks and mitigations

- Single point of failure (DNS/DHCP node): mitigate with keepalived (VRRP) to float 192.168.1.53 to `rpi2` when `rpi1` fails.
- Device DNS stickiness: some devices cache DNS aggressively; use service reloads or power-cycle in worst case. Prefer static leases with short TTLs during transition.
- Private key replication: default to management key only; keep the “shared key on all nodes” as an optional opt-in.

### What “good” looks like

- Toggling between home and edge is a single command that completes in seconds.
- All nodes resolve each other via `*.tatbot.local`; MCP servers are reachable at stable hostnames and ports.
- Home mode: devices integrate with the home network and are accessible from `oop`.
- Edge mode: the system forms a complete, isolated network that continues to operate without the home router.
- Security posture improved: fewer secrets spread, restricted inbound surfaces, validated changes, and simple rollback.


