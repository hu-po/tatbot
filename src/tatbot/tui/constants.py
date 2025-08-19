"""Constants for TUI monitoring system."""

# Default refresh rate for TUI updates (in seconds)
DEFAULT_REFRESH_RATE = 2.0

# Network timeouts
MCP_HEALTH_CHECK_TIMEOUT = 2.0
SSH_HEALTH_CHECK_TIMEOUT = 1.0

# Node IP addresses
NODE_IPS = {
    "eek": "192.168.1.97",
    "hog": "192.168.1.98", 
    "ook": "192.168.1.99",
    "oop": "192.168.1.51",
    "ojo": "192.168.1.100",
    "rpi1": "192.168.1.93",
    "rpi2": "192.168.1.94"
}

# Node roles for display
NODE_ROLES = {
    "eek": "redis",
    "hog": "robot",
    "ook": "ik,viz", 
    "ojo": "agent,policy",
    "rpi1": "tui",
    "rpi2": "dns",
    "oop": "dev",
}