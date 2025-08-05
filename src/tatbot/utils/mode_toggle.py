import logging
import os
from typing import List, Literal, Tuple

import paramiko
import yaml

from tatbot.data.node import Node
from tatbot.utils.log import get_logger, setup_log_with_config

log = get_logger("toggle_dns", "🔀")

class ToggleDNSConfig:
    debug: bool = False
    """Enable debug logging."""
    
    config_path: str = os.path.expanduser("~/tatbot/src/conf/nodes.yaml")
    """Path to nodes config YAML."""
    
    dns_node_name: str = "rpi2"
    """Name of the DNS server node."""
    
    router_dns: str = "192.168.1.1"
    """DNS server in edge mode (LAN router)."""
    
    domain: str = "tatbot.local"
    """Local domain for DNS."""
    
    mode: Literal["home", "edge", "toggle"] = "toggle"
    """Mode to switch to (default: toggle)"""

class NetworkToggler:
    """Toggle DNS configuration between home (DNS node) and edge (LAN router DNS) modes."""

    def __init__(self, config: ToggleDNSConfig):
        self.config = config
        self.nodes = self._load_nodes()
        self.key_path = os.path.expanduser("~/.ssh/tatbot-key")  # Use the existing key
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
            # Use SSH key for authentication
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
        _, stdout, stderr = client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode().strip(), stderr.read().decode().strip()

    def _update_node_dns(self, mode: str):
        dns_server = self.dns_node.ip if mode == "home" else ""
        for node in self.nodes:
            if node.name == self.config.dns_node_name: 
                continue
            if node.name == "oop":
                continue

            log.info(f"Updating DNS for {node.name} to {'home' if dns_server else 'edge'} mode...")
            client = self._get_ssh_client(node)

            if dns_server:
                # Add static DNS for home mode
                cmd = f"sudo sh -c 'grep -q \"static domain_name_servers\" /etc/dhcpcd.conf || echo \"static domain_name_servers={dns_server}\" >> /etc/dhcpcd.conf'"
            else:
                # Remove static DNS for edge mode
                cmd = "sudo sed -i '/^static domain_name_servers/d' /etc/dhcpcd.conf"
            
            exit_code, out, err = self._run_remote(client, cmd)
            if exit_code != 0:
                log.error(f"Failed to update DNS config on {node.name}: {err}")
            else:
                # Restart service to apply changes
                restart_cmd = "sudo systemctl restart dhcpcd"
                exit_code, out, err = self._run_remote(client, restart_cmd)
                if exit_code != 0:
                    log.warning(f"Failed to restart dhcpcd on {node.name}: {err}")
            
            client.close()

    def _update_arm_dns_config(self, mode: str):
        dns_ip = self.dns_node.ip if mode == "home" else self.config.router_dns
        log.info(f"Updating arm YAML configs with DNS: {dns_ip}")
        
        for arm in ['l', 'r']:
            filepath = os.path.expanduser(f"~/tatbot/src/conf/trossen/arm_{arm}.yaml")
            try:
                with open(filepath, 'r') as f:
                    config = yaml.safe_load(f)
                
                config['dns'] = dns_ip
                
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                log.info(f"Updated {filepath} with DNS: {dns_ip}")
            except Exception as e:
                log.error(f"Failed to update {filepath}: {e}")

    def to_home(self):
        log.info("Switching to HOME mode...")
        
        # 1. Update arm config files
        self._update_arm_dns_config("home")

        # 2. Start DNS on DNS node
        client = self._get_ssh_client(self.dns_node)
        cmds = [
            "sudo systemctl start dnsmasq",
            "sudo systemctl enable dnsmasq"
        ]
        for cmd in cmds:
            exit_code, out, err = self._run_remote(client, cmd)
            if exit_code != 0:
                log.error(f"Failed on {self.config.dns_node_name}: {cmd} - {err}")
        client.close()

        # 3. Update other nodes' DNS persistently
        self._update_node_dns("home")
        log.info("✅ Switched to HOME mode.")

    def to_edge(self):
        log.info("Switching to EDGE mode...")
        
        # 1. Update arm config files
        self._update_arm_dns_config("edge")

        # 2. Stop DNS on DNS node
        client = self._get_ssh_client(self.dns_node)
        cmds = [
            "sudo systemctl stop dnsmasq",
            "sudo systemctl disable dnsmasq"
        ]
        for cmd in cmds:
            exit_code, out, err = self._run_remote(client, cmd)
            if exit_code != 0:
                log.error(f"Failed on {self.config.dns_node_name}: {cmd} - {err}")
        client.close()

        # 3. Update other nodes' DNS persistently
        self._update_node_dns("edge")
        log.info("✅ Switched to EDGE mode.")

    def toggle(self):
        # Detect current mode by checking dhcpcd.conf on a non-DNS node
        test_node = next((n for n in self.nodes if n.name != self.config.dns_node_name), None)
        if not test_node:
            log.error("No test node found")
            return
        
        client = self._get_ssh_client(test_node)
        _, out, _ = self._run_remote(client, "grep -q 'static domain_name_servers' /etc/dhcpcd.conf && echo 'home' || echo 'edge'")
        client.close()
        
        if "home" in out:
            self.to_edge()
        else:
            self.to_home()

if __name__ == "__main__":
    config = setup_log_with_config(ToggleDNSConfig)
    if config.debug:
        log.setLevel(logging.DEBUG)
    
    toggler = NetworkToggler(config)
    
    if config.mode == "home":
        toggler.to_home()
    elif config.mode == "edge":
        toggler.to_edge()
    else:
        toggler.toggle()