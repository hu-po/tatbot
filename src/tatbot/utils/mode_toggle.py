#!/usr/bin/env python3
"""
Centralized DNS mode toggler for tatbot network.
Uses dnsmasq configuration profiles on rpi2 for fast, reliable mode switching.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Literal, Tuple
import tempfile

import paramiko
import yaml

from tatbot.data.node import Node
from tatbot.utils.log import get_logger, setup_log_with_config
from tatbot.utils.network_config import NetworkConfig

log = get_logger("mode_toggle", "🔀")

class CentralizedDNSConfig:
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

class CentralizedNetworkToggler:
    """Centralized DNS mode toggler using dnsmasq configuration profiles."""

    def __init__(self, config: CentralizedDNSConfig):
        self.config = config
        self.nodes = self._load_nodes()
        self.key_path = os.path.expanduser("~/.ssh/tatbot-key")
        self.dns_node = next((n for n in self.nodes if n.name == self.config.dns_node_name), None)
        if not self.dns_node:
            raise ValueError(f"{self.config.dns_node_name} not found in nodes config")
        
        # Initialize network config generator (lazy loading)
        self.network_config = NetworkConfig()
        self._configs_loaded = False

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

    def _ensure_configs_loaded(self):
        """Ensure network configs are loaded from ip_addresses_dump.md."""
        if not self._configs_loaded:
            try:
                self.network_config.parse_ip_dump()
                self._configs_loaded = True
            except FileNotFoundError:
                log.error("ip_addresses_dump.md not found. Run network_config.py first to generate it.")
                raise
    
    def _deploy_dnsmasq_configs(self) -> bool:
        """Deploy dnsmasq configuration files to DNS node."""
        log.info(f"Deploying dnsmasq configs to {self.config.dns_node_name}...")
        
        # Ensure configs are loaded
        self._ensure_configs_loaded()
        
        client = self._get_ssh_client(self.dns_node)
        
        try:
            # Create dnsmasq directories if they don't exist
            for directory in [self.config.dnsmasq_config_dir, self.config.dnsmasq_profiles_dir]:
                cmd = f"sudo mkdir -p {directory}"
                exit_code, _, err = self._run_remote(client, cmd)
                if exit_code != 0:
                    log.error(f"Failed to create directory {directory}: {err}")
                    return False
            
            # Generate and upload configurations
            configs = {
                "mode-home.conf": self.network_config.generate_dnsmasq_home_config(),
                "mode-edge.conf": self.network_config.generate_dnsmasq_edge_config()
            }
            
            for filename, content in configs.items():
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    # Copy to remote host
                    sftp = client.open_sftp()
                    remote_tmp = f"/tmp/{filename}"
                    sftp.put(tmp_path, remote_tmp)
                    sftp.close()
                    
                    # Move to profiles directory with sudo
                    cmd = f"sudo mv {remote_tmp} {self.config.dnsmasq_profiles_dir}/{filename}"
                    exit_code, _, err = self._run_remote(client, cmd)
                    if exit_code != 0:
                        log.error(f"Failed to deploy {filename}: {err}")
                        return False
                    
                    log.info(f"Deployed {filename} to profiles directory")
                    
                finally:
                    # Clean up local temp file
                    os.unlink(tmp_path)
            
            return True
            
        finally:
            client.close()

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
        """Verify that mode switch was successful."""
        log.info("Verifying mode switch...")
        
        # Check that active symlink points to correct config
        actual_mode = self._get_current_mode()
        if actual_mode != expected_mode:
            log.error(f"Mode verification failed: expected {expected_mode}, got {actual_mode}")
            return False
        
        # Check that dnsmasq is running
        client = self._get_ssh_client(self.dns_node)
        try:
            cmd = "sudo systemctl is-active dnsmasq"
            exit_code, out, _ = self._run_remote(client, cmd)
            if exit_code != 0 or "active" not in out:
                log.error("dnsmasq service is not active")
                return False
        finally:
            client.close()
        
        # Direct DNS resolution test against rpi2
        test_host = f"{self.dns_node.name}.{self.config.domain}"
        try:
            import subprocess
            cmd = f"nslookup {test_host} {self.dns_node.ip}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                log.info(f"DNS resolution test passed: {test_host}")
            else:
                log.warning(f"DNS resolution test failed for {test_host}")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            log.warning(f"DNS resolution test failed: {e}")
        
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
        
        if not self._deploy_dnsmasq_configs():
            return False
        
        # Setup dnsmasq configuration and initial profile
        client = self._get_ssh_client(self.dns_node)
        try:
            active_path = f"{self.config.dnsmasq_config_dir}/active.conf"
            home_path = f"{self.config.dnsmasq_profiles_dir}/mode-home.conf"
            
            # Configure dnsmasq to use only the active config file
            dnsmasq_conf = f"conf-file={active_path}"
            cmd = f"echo '{dnsmasq_conf}' | sudo tee /etc/dnsmasq.conf"
            exit_code, _, err = self._run_remote(client, cmd)
            if exit_code != 0:
                log.error(f"Failed to configure dnsmasq.conf: {err}")
                return False
            
            # Only create symlink if doesn't exist
            cmd = f"test -L {active_path} || sudo ln -s {home_path} {active_path}"
            self._run_remote(client, cmd)
            
            # Enable and start dnsmasq service
            cmd = "sudo systemctl enable dnsmasq"
            self._run_remote(client, cmd)
            
            cmd = "sudo systemctl start dnsmasq"
            exit_code, _, err = self._run_remote(client, cmd)
            if exit_code != 0:
                log.warning(f"Failed to start dnsmasq: {err}")
            
        finally:
            client.close()
        
        log.info("DNS control node setup complete")
        return True

    def to_home(self) -> bool:
        """Switch to home mode (DNS forwarder)."""
        log.info("Switching to HOME mode...")
        
        success = self._switch_dnsmasq_profile("home")
        if success and self.config.validate:
            success = self._verify_mode_switch("home")
        
        if success:
            log.info("✅ Successfully switched to HOME mode")
        else:
            log.error("❌ Failed to switch to HOME mode")
        
        return success

    def to_edge(self) -> bool:
        """Switch to edge mode (authoritative DNS + DHCP)."""
        log.info("Switching to EDGE mode...")
        
        success = self._switch_dnsmasq_profile("edge")
        if success and self.config.validate:
            success = self._verify_mode_switch("edge")
        
        if success:
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
            log.warning(f"Unknown current mode: {current_mode}, defaulting to home")
            return self.to_home()

    def status(self) -> str:
        """Get current mode status."""
        current_mode = self._get_current_mode()
        
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
            
        finally:
            client.close()
        
        status_info = {
            "mode": current_mode,
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
    
    config = CentralizedDNSConfig()
    config.mode = args.mode
    config.debug = args.debug
    config.validate = not args.no_validate
    
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    toggler = CentralizedNetworkToggler(config)
    
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