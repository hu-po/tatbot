#!/usr/bin/env python3
"""
Simple DNS mode toggler for tatbot network.
Switches between edge mode (local DNS+DHCP) and home mode (DNS forwarding).
"""

import logging
import os
from typing import List, Literal, Tuple

import paramiko
import yaml

from tatbot.data.node import Node
from tatbot.utils.log import get_logger

log = get_logger("mode_toggle", "🔀")

class DNSConfig:
    debug: bool = False
    """Enable debug logging."""
    
    config_path: str = os.path.expanduser("~/tatbot/src/conf/nodes.yaml")
    """Path to nodes config YAML."""
    
    dns_node_name: str = "rpi2"
    """Name of the DNS control node."""
    
    domain: str = "tatbot.lan"
    """Local domain for DNS (avoid .local which conflicts with mDNS)."""
    
    mode: Literal["home", "edge", "toggle", "status"] = "toggle"
    """Mode to switch to or check"""
    
    dnsmasq_config_dir: str = "/etc/dnsmasq.d"
    """Directory for dnsmasq configuration files on DNS node"""
    
    dnsmasq_profiles_dir: str = "/etc/dnsmasq-profiles"
    """Directory for dnsmasq mode profiles on DNS node"""
    
    validate: bool = True
    """Validate configuration before applying"""

class NetworkToggler:
    """Simple DNS mode toggler using dnsmasq configuration profiles."""

    def __init__(self, config: DNSConfig):
        self.config = config
        self.nodes = self._load_nodes()
        self.key_path = os.path.expanduser("~/.ssh/tatbot-key")
        self.dns_node = next((n for n in self.nodes if n.name == self.config.dns_node_name), None)
        if not self.dns_node:
            raise ValueError(f"{self.config.dns_node_name} not found in nodes config")

    def _load_nodes(self) -> List[Node]:
        with open(self.config.config_path, "r") as f:
            nodes_data = yaml.safe_load(f)["nodes"]
        return [Node(**n) for n in nodes_data]

    def _get_ssh_client(self, node: Node) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                node.ip,
                username=node.user,
                key_filename=self.key_path,
                timeout=10.0
            )
            return client
        except Exception as e:
            log.error(f"Failed to connect to {node.name} using key: {e}")
            raise

    def _run_remote(self, client: paramiko.SSHClient, cmd: str) -> Tuple[int, str, str]:
        """Execute command on remote host and return exit code, stdout, stderr."""
        _, stdout, stderr = client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode().strip(), stderr.read().decode().strip()

    def _validate_dnsmasq_config(self, config_file: str) -> bool:
        """Validate dnsmasq configuration file."""
        if not self.config.validate:
            return True
            
        log.info(f"Validating dnsmasq configuration: {config_file}")
        client = self._get_ssh_client(self.dns_node)
        
        try:
            cmd = f"sudo dnsmasq --test --conf-file={config_file}"
            exit_code, out, err = self._run_remote(client, cmd)
            
            if exit_code == 0:
                log.info("Configuration validation passed")
                return True
            else:
                log.error(f"Configuration validation failed: {err}")
                return False
                
        finally:
            client.close()

    def _switch_dnsmasq_profile(self, mode: str) -> bool:
        """Switch dnsmasq configuration profile using symlink."""
        config_file = f"mode-{mode}.conf"
        config_path = f"{self.config.dnsmasq_profiles_dir}/{config_file}"
        active_path = f"{self.config.dnsmasq_config_dir}/active.conf"
        
        log.info(f"Switching to {mode} mode profile...")
        
        client = self._get_ssh_client(self.dns_node)
        
        try:
            # Validate the target configuration
            if not self._validate_dnsmasq_config(config_path):
                return False
            
            # Create backup of current active config
            backup_cmd = f"sudo cp {active_path} {active_path}.backup 2>/dev/null || true"
            self._run_remote(client, backup_cmd)
            
            # Switch symlink
            switch_cmd = f"sudo ln -sf {config_path} {active_path}"
            exit_code, _, err = self._run_remote(client, switch_cmd)
            if exit_code != 0:
                log.error(f"Failed to switch symlink: {err}")
                return False
            
            # Reload dnsmasq
            reload_cmd = "sudo systemctl reload dnsmasq"
            exit_code, _, err = self._run_remote(client, reload_cmd)
            if exit_code != 0:
                log.error(f"Failed to reload dnsmasq: {err}")
                # Try to restore backup
                restore_cmd = f"sudo mv {active_path}.backup {active_path} 2>/dev/null || true"
                self._run_remote(client, restore_cmd)
                return False
            
            log.info(f"Successfully switched to {mode} mode")
            return True
            
        finally:
            client.close()

    def _check_home_router(self) -> bool:
        """Check if home router is reachable from DNS node."""
        client = self._get_ssh_client(self.dns_node)
        
        try:
            # Ping home router with short timeout
            cmd = "ping -c 1 -W 2 192.168.1.1"
            exit_code, _, _ = self._run_remote(client, cmd)
            return exit_code == 0
        finally:
            client.close()
    
    def _get_current_mode(self) -> str:
        """Detect current mode by checking active dnsmasq configuration."""
        client = self._get_ssh_client(self.dns_node)
        
        try:
            active_path = f"{self.config.dnsmasq_config_dir}/active.conf"
            cmd = f"readlink {active_path}"
            exit_code, out, err = self._run_remote(client, cmd)
            
            if exit_code == 0:
                if "mode-home.conf" in out:
                    return "home"
                elif "mode-edge.conf" in out:
                    return "edge" 
                else:
                    return "unknown"
            else:
                # Check if dnsmasq is running to guess mode
                cmd = "sudo systemctl is-active dnsmasq"
                exit_code, out, _ = self._run_remote(client, cmd)
                if exit_code == 0 and "active" in out:
                    return "unknown_active"
                else:
                    return "unknown_inactive"
                    
        finally:
            client.close()

    def _verify_mode_switch(self, expected_mode: str) -> bool:
        """Basic verification that mode switch was successful."""
        log.info("Verifying mode switch...")
        
        # Check that active symlink points to correct config
        actual_mode = self._get_current_mode()
        if actual_mode != expected_mode:
            log.error(f"Mode verification failed: expected {expected_mode}, got {actual_mode}")
            return False
        
        log.info("Mode switch verification passed")
        return True

    def _ensure_dnsmasq_installed(self) -> bool:
        """Ensure dnsmasq is installed on DNS node."""
        log.info("Checking dnsmasq installation...")
        
        client = self._get_ssh_client(self.dns_node)
        try:
            # Check if dnsmasq is installed
            cmd = "which dnsmasq"
            exit_code, _, _ = self._run_remote(client, cmd)
            
            if exit_code == 0:
                log.info("dnsmasq is already installed")
                return True
            
            # Install dnsmasq
            log.info("Installing dnsmasq...")
            cmd = "sudo apt update && sudo apt install -y dnsmasq"
            exit_code, out, err = self._run_remote(client, cmd)
            
            if exit_code == 0:
                log.info("Successfully installed dnsmasq")
                return True
            else:
                log.error(f"Failed to install dnsmasq: {err}")
                return False
                
        finally:
            client.close()

    def setup(self) -> bool:
        """Set up DNS node with required configurations."""
        log.info("Setting up DNS control node...")
        
        if not self._ensure_dnsmasq_installed():
            return False
        
        # Setup dnsmasq configuration and initial profile (default to edge mode)
        client = self._get_ssh_client(self.dns_node)
        try:
            active_path = f"{self.config.dnsmasq_config_dir}/active.conf"
            edge_path = f"{self.config.dnsmasq_profiles_dir}/mode-edge.conf"
            
            # Configure dnsmasq to use only the active config file
            dnsmasq_conf = f"conf-file={active_path}"
            cmd = f"echo '{dnsmasq_conf}' | sudo tee /etc/dnsmasq.conf"
            exit_code, _, err = self._run_remote(client, cmd)
            if exit_code != 0:
                log.error(f"Failed to configure dnsmasq.conf: {err}")
                return False
            
            # Create symlink to edge mode (default on boot)
            cmd = f"sudo ln -sf {edge_path} {active_path}"
            self._run_remote(client, cmd)
            
            # Enable and start dnsmasq service
            cmd = "sudo systemctl enable dnsmasq && sudo systemctl start dnsmasq"
            exit_code, _, err = self._run_remote(client, cmd)
            if exit_code != 0:
                log.warning(f"Failed to start dnsmasq: {err}")
            
        finally:
            client.close()
        
        log.info("DNS control node setup complete (default: edge mode)")
        return True

    def to_home(self) -> bool:
        """Switch to home mode (DNS forwarder to home router)."""
        log.info("Switching to HOME mode...")
        
        success = self._switch_dnsmasq_profile("home")
        if success:
            log.info("✅ Successfully switched to HOME mode")
        else:
            log.error("❌ Failed to switch to HOME mode")
        
        return success

    def to_edge(self) -> bool:
        """Switch to edge mode (local DNS + conditional DHCP)."""
        log.info("Switching to EDGE mode...")
        
        success = self._switch_dnsmasq_profile("edge")
        
        if success:
            # Check if home router is present and conditionally disable DHCP
            if self._check_home_router():
                log.info("Home router detected - disabling DHCP in edge mode to prevent conflicts")
                client = self._get_ssh_client(self.dns_node)
                try:
                    # Comment out dhcp-range to disable DHCP
                    cmd = "sudo sed -i 's/^dhcp-range=/#dhcp-range=/' /etc/dnsmasq-profiles/mode-edge.conf"
                    self._run_remote(client, cmd)
                    # Reload dnsmasq
                    self._run_remote(client, "sudo systemctl reload dnsmasq")
                finally:
                    client.close()
            else:
                log.info("No home router detected - DHCP enabled in edge mode")
                client = self._get_ssh_client(self.dns_node)
                try:
                    # Uncomment dhcp-range to enable DHCP
                    cmd = "sudo sed -i 's/^#dhcp-range=/dhcp-range=/' /etc/dnsmasq-profiles/mode-edge.conf"
                    self._run_remote(client, cmd)
                    # Reload dnsmasq
                    self._run_remote(client, "sudo systemctl reload dnsmasq")
                finally:
                    client.close()
            
            log.info("✅ Successfully switched to EDGE mode")
        else:
            log.error("❌ Failed to switch to EDGE mode")
        
        return success

    def toggle(self) -> bool:
        """Toggle between home and edge modes."""
        current_mode = self._get_current_mode()
        log.info(f"Current mode: {current_mode}")
        
        if current_mode == "home":
            return self.to_edge()
        elif current_mode == "edge":
            return self.to_home()
        else:
            log.warning(f"Unknown current mode: {current_mode}, defaulting to edge (default)")
            return self.to_edge()

    def status(self) -> str:
        """Get current mode status."""
        current_mode = self._get_current_mode()
        home_router_reachable = self._check_home_router()
        
        # Get additional status info
        client = self._get_ssh_client(self.dns_node)
        try:
            # Check dnsmasq status
            cmd = "sudo systemctl status dnsmasq --no-pager -l"
            exit_code, out, _ = self._run_remote(client, cmd)
            dnsmasq_status = "active" if exit_code == 0 and "active (running)" in out else "inactive"
            
            # Get active config file
            active_path = f"{self.config.dnsmasq_config_dir}/active.conf"
            cmd = f"readlink {active_path} 2>/dev/null || echo 'no symlink'"
            _, config_link, _ = self._run_remote(client, cmd)
            
            # Check if DHCP is enabled in edge mode
            dhcp_enabled = "unknown"
            if current_mode == "edge":
                cmd = "grep '^dhcp-range=' /etc/dnsmasq-profiles/mode-edge.conf"
                exit_code, _, _ = self._run_remote(client, cmd)
                dhcp_enabled = "yes" if exit_code == 0 else "no"
            
        finally:
            client.close()
        
        status_info = {
            "mode": current_mode,
            "home_router_reachable": home_router_reachable,
            "dhcp_enabled": dhcp_enabled if current_mode == "edge" else "n/a",
            "dnsmasq_status": dnsmasq_status,
            "active_config": config_link,
            "dns_node": self.config.dns_node_name,
            "dns_node_ip": self.dns_node.ip
        }
        
        return yaml.dump(status_info, default_flow_style=False)

def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Centralized DNS mode toggler for tatbot")
    parser.add_argument("--mode", choices=["home", "edge", "toggle", "status"], 
                       default="toggle", help="Mode to switch to")
    parser.add_argument("--setup", action="store_true", 
                       help="Set up DNS node with configurations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-validate", action="store_true", 
                       help="Skip configuration validation")
    
    args = parser.parse_args()
    
    config = DNSConfig()
    config.mode = args.mode
    config.debug = args.debug
    config.validate = not args.no_validate
    
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    toggler = NetworkToggler(config)
    
    if args.setup:
        success = toggler.setup()
        exit(0 if success else 1)
    
    if config.mode == "home":
        success = toggler.to_home()
    elif config.mode == "edge":
        success = toggler.to_edge()
    elif config.mode == "status":
        print(toggler.status())
        exit(0)
    else:
        success = toggler.toggle()
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()