# Network Architecture Refactor: Security & Convenience Analysis

## Current Network Situation

### Architecture Overview
The tatbot system currently operates in two distinct modes:
- **Home Mode**: All devices connect through home router (192.168.1.x), with `oop` available
- **Edge Mode**: Standalone network with `rpi2` as DNS server, no external connectivity

### Current Pain Points

#### Security Concerns
1. **Shared SSH Keys**: Single `tatbot-key` distributed to all nodes
   - No granular access control
   - Key compromise affects entire system
   - No key rotation mechanism

2. **Open Network Access**: All devices visible on home network
   - IP cameras accessible from any home device
   - Robot arms controllable from any network client
   - No network segmentation or access controls

3. **Credential Management**: Hardcoded passwords in `.env` files
   - Camera admin credentials in plaintext
   - No secure credential storage
   - Credentials shared across multiple systems

#### Convenience Issues
1. **Mode Switching Complexity**: 
   - Manual SSH to each device
   - Sequential configuration changes (30+ seconds)
   - No atomic transactions or rollback capability
   - Requires deep knowledge of each device type

2. **Service Discovery**: 
   - Hardcoded IP addresses in configurations
   - No automatic MCP server discovery
   - Manual port management across nodes

3. **Heterogeneous Device Management**:
   - RPi uses dhcpcd, Jetson uses netplan, cameras use web API
   - Different restart procedures per device type
   - Inconsistent configuration patterns

## Proposed Secure & Convenient Architecture

### Core Design Principles
1. **Zero-Trust Network**: Verify all communications regardless of network location
2. **Least Privilege Access**: Each component has minimal required permissions
3. **Defense in Depth**: Multiple security layers with graceful degradation
4. **Operational Simplicity**: Single-command mode switching with automatic verification

### Network Topology Redesign

#### Edge Mode: Secure Isolated Network
```
┌─────────────────────────────────────────────────────────────┐
│ Edge Mode (192.168.100.0/24)                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  rpi2 (100.1) ─┐                                          │
│  DNS/DHCP/CA    │ ┌─ eek (100.10) ─ arm-l/arm-r    │
│                 ├─┤                                        │
│  rpi1 (100.2) ──┘ ├─ ojo (100.20) ─ realsense1/2         │
│  mTLS Gateway      │                                        │
│                    ├─ ook (100.30)                         │
│                    │                                        │
│                    └─ camera1-5 (100.40-44)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Home Mode: Bridged Access
```
┌───────────────────────────────────────────────────────────────┐
│ Home Network (192.168.1.0/24)                                │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Home Router (1.1) ─┬─ oop (1.51) ← Developer Access        │
│                     │                                         │
│                     ├─ tatbot-bridge (1.100)                 │
│                     │   ↓ mTLS Tunnel                        │
│                     │                                         │
│  ┌──────────────────┼─────────────────────────────────────┐   │
│  │ Tatbot Subnet (1.90-99)                              │   │
│  │                  │                                    │   │
│  │  rpi2 (1.99) ────┤  ook (1.90)                       │   │
│  │  rpi1 (1.98) ────┤  ojo (1.96)                       │   │
│  │  eek ─────┤  cameras (1.101-105)              │   │
│  │  (1.97)          │                                    │   │
│  └──────────────────┼─────────────────────────────────────┘   │
│                     │                                         │
└───────────────────────────────────────────────────────────────┘
```

### Security Architecture

#### 1. Mutual TLS (mTLS) Communication
```yaml
# PKI Infrastructure
root_ca: rpi2
intermediate_ca: rpi1  # For device certificates
client_certs:
  - ook.tatbot.local
  - ojo.tatbot.local
  - eek.tatbot.local
service_certs:
  - mcp.tatbot.local
  - camera-api.tatbot.local
```

**Implementation:**
- Each node gets unique client certificate
- MCP servers require valid client certificates
- Automatic certificate rotation (30-day lifecycle)
- Certificate pinning for critical connections

#### 2. Network Segmentation
```python
class NetworkZones:
    CONTROL_PLANE = "control"     # MCP servers, coordination
    DATA_PLANE = "data"           # Camera streams, sensor data  
    ROBOT_PLANE = "robot"         # Arm controllers, actuators
    MANAGEMENT = "mgmt"           # SSH, monitoring, updates
```

**VLANs/Subnets:**
- Control traffic isolated from data streams
- Robot commands on dedicated network segment
- Management access through secure gateway only

#### 3. Service Mesh with Authentication
```python
# MCP Service Registry with Auth
class SecureMCPServer:
    def __init__(self):
        self.cert_manager = CertificateManager()
        self.auth_policy = AuthorizationPolicy()
        self.rate_limiter = RateLimiter()
    
    def handle_request(self, request):
        # 1. Verify client certificate
        # 2. Check authorization policy
        # 3. Apply rate limiting
        # 4. Execute with least privilege
```

### Convenience Features

#### 1. Single-Command Mode Switching
```bash
# Atomic mode switch with verification
tatbot network-mode edge --verify --timeout 30s

# Status and health check
tatbot network-status --all-services --security-check

# Emergency access (bypasses normal auth for troubleshooting)
tatbot emergency-access --node eek --reason "arm-stuck"
```

#### 2. Automatic Service Discovery
```python
class ServiceDiscovery:
    def __init__(self):
        self.consul_client = ConsulClient()
        self.dns_resolver = SecureDNSResolver()
    
    def discover_mcp_servers(self) -> List[MCPEndpoint]:
        # Auto-discover via mDNS + Consul
        # Return authenticated endpoints with health status
        # Cache with TTL for performance
```

#### 3. Device Profile Management
```yaml
# Device capability auto-detection
device_profiles:
  rpi:
    capabilities: [dns_server, nfs_server, certificate_authority]
    config_methods: [systemd, dhcpcd]
    security_features: [ssh_keys, certificates, firewall]
  
  jetson:
    capabilities: [inference, vision_processing]
    config_methods: [netplan, systemd]
    security_features: [secure_boot, tpm, certificates]
    
  trossen_controller:
    capabilities: [robot_control, kinematics]
    config_methods: [yaml_config, rest_api]
    security_features: [api_keys, rate_limiting]
```

### Implementation Plan

#### Phase 1: Certificate Infrastructure (Week 1)
1. **Root CA Setup**: Configure rpi2 as certificate authority
2. **Certificate Generation**: Create node and service certificates
3. **mTLS Integration**: Update MCP servers for certificate auth
4. **Testing**: Verify encrypted communication between all nodes

#### Phase 2: Network Segmentation (Week 2)
1. **VLAN Configuration**: Setup logical network separation
2. **Firewall Rules**: Implement least-privilege network policies
3. **Service Discovery**: Deploy Consul for automatic service registration
4. **Gateway Setup**: Configure secure access gateway for home mode

#### Phase 3: Mode Switching Automation (Week 3)
1. **Device Abstraction**: Implement device profile system
2. **Transaction Engine**: Build atomic configuration change system
3. **Health Monitoring**: Add comprehensive system health checks
4. **CLI Interface**: Create user-friendly command interface

#### Phase 4: Security Hardening (Week 4)
1. **Credential Rotation**: Implement automatic key/certificate rotation
2. **Audit Logging**: Add security event logging and monitoring
3. **Intrusion Detection**: Deploy network anomaly detection
4. **Emergency Procedures**: Create secure emergency access methods

### Security Benefits

1. **Authentication**: Every connection verified with certificates
2. **Authorization**: Fine-grained access control per service
3. **Confidentiality**: All traffic encrypted end-to-end
4. **Integrity**: Message signing prevents tampering
5. **Availability**: Redundant services with failover capability
6. **Auditability**: Complete security event logging

### Convenience Benefits

1. **Single Command**: `tatbot network-mode edge` replaces complex SSH procedures
2. **Automatic Discovery**: No hardcoded IPs, services self-register
3. **Health Monitoring**: Continuous system status with automatic alerts
4. **Rollback Safety**: Failed changes automatically reverted
5. **Zero Downtime**: Hot switching between modes without service interruption

### Edge Mode Network Behavior

In edge mode, the system creates a complete isolated network:

```
┌─────────────────────────────────────────────────────────────┐
│ Internal Services (No External Access)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ • DNS: rpi2 resolves *.tatbot.local internally            │
│ • DHCP: rpi2 assigns 192.168.100.x addresses              │
│ • NFS: rpi2 serves shared storage to all nodes            │
│ • mTLS: All MCP communication encrypted & authenticated    │
│ • Service Discovery: Automatic MCP server registration     │
│                                                             │
│ Node Communication:                                         │
│ • ook ↔ eek: GPU batch processing requests         │
│ • ojo ↔ cameras: Vision processing streams                │
│ • All nodes ↔ MCP registry: Service coordination          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Home Mode Network Behavior

In home mode, tatbot integrates seamlessly with your home network:

```
┌─────────────────────────────────────────────────────────────┐
│ Home Integration (Secure Bridge)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ • Developer Access: oop can reach all tatbot services      │
│ • Secure Gateway: mTLS tunnel protects internal traffic    │
│ • Home DNS: Router resolves external domains              │
│ • Internet Access: All nodes can reach external services   │
│                                                             │
│ Access Patterns:                                            │
│ • oop → tatbot services (via secure gateway)              │
│ • tatbot nodes → internet (via home router)                │
│ • Internal tatbot communication (via mTLS mesh)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Conclusion

This refactored architecture provides enterprise-grade security while maintaining the operational simplicity required for a robotics research system. The key innovation is using mTLS service mesh for zero-trust communication combined with intelligent automation for seamless mode switching.

The system becomes both more secure (defense in depth, least privilege) and more convenient (single-command operations, automatic service discovery) than the current implementation.