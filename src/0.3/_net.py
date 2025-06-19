import concurrent.futures
from dataclasses import dataclass
import getpass
import logging
import os
import socket
import subprocess
from typing import Optional, Union, Tuple

import paramiko
from paramiko.client import SSHClient
from paramiko.sftp_client import SFTPClient
import yaml

from _log import get_logger, setup_log_with_config, print_config

log = get_logger('_net')

@dataclass
class SetupNetConfig:
    debug: bool = False
    """Enable debug logging."""
    shared_key_name: str = "tatbot-key"
    ssh_dir: str = os.path.expanduser("~/.ssh")
    key_path: str = os.path.join(ssh_dir, shared_key_name)
    pub_key_path: str = f"{key_path}.pub"
    config_path: str = os.path.join(ssh_dir, "config")
    yaml_file: str = os.path.expanduser("~/tatbot/config/nodes.yaml")
    test_timeout: float = 5.0
    """Timeout in seconds for node connectivity tests."""


def run(command, **kwargs):
    log.info(f"🌐 Running: {command}")
    subprocess.run(command, shell=True, check=True, **kwargs)

def generate_key(config: SetupNetConfig):
    if not os.path.exists(config.key_path):
        log.info(f"🌐 Generating SSH key at {config.key_path}...")
        run(f"ssh-keygen -t rsa -b 4096 -f {config.key_path} -N ''")
    else:
        log.debug(f"🌐 SSH key already exists at {config.key_path}. Skipping generation.")

def load_nodes(yaml_file: str):
    log.info(f"🌐 Loading nodes from {yaml_file}")
    with open(yaml_file, "r") as f:
        nodes = yaml.safe_load(f)["nodes"]
    log.debug(f"🌐 Loaded nodes: {nodes}")
    return nodes

def _get_local_ips() -> set[str]:
    """Return a set of IP addresses assigned to this host.

    Uses socket utilities only to avoid external dependencies.
    """
    ips: set[str] = set()
    hostname = socket.gethostname()

    # Add addresses resolved via hostname
    try:
        ips.update(socket.gethostbyname_ex(hostname)[2])
    except socket.gaierror:
        pass

    # Add addresses gathered from all interfaces
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
        ips.update(ai[4][0] for ai in addrinfo)
    except socket.gaierror:
        pass

    # Common loopback addresses
    ips.update({"127.0.0.1", "0.0.0.0"})
    return ips

# Pre-compute to avoid repeated system calls
_LOCAL_IPS = _get_local_ips()
_LOCAL_HOSTNAMES = {socket.gethostname(), socket.getfqdn(), "localhost"}

def _is_local_node(node) -> bool:
    """Return True if the node represents this host."""
    return node.get("ip") in _LOCAL_IPS or node.get("name") in _LOCAL_HOSTNAMES

def distribute_keys(nodes, config: SetupNetConfig):
    """Copy SSH keys to each remote node and authorize the public key.

    Uses a single Paramiko SSH connection per node, so the user only needs to
    enter their password once for each node (or not at all if the key already
    works).
    """
    password_cache: dict[str, str] = {}

    for node in nodes:
        if _is_local_node(node):
            log.debug(f"🛑 Skipping key distribution for local node {node['name']} ({node['ip']})")
            continue

        name, ip, user, emoji = (
            node["name"],
            node["ip"],
            node["user"],
            node.get("emoji", "🌐"),
        )

        log.info(f"{emoji} Setting up {name} ({ip})")

        # Try key-based auth first (may already be configured from previous run)
        try:
            client = get_ssh_client(ip, user, config.key_path)
            log.debug(f"{emoji} Connected to {name} with existing key – no password needed 🎉")
            need_password = False
        except Exception:
            need_password = True

        if need_password:
            pw = password_cache.get(user)
            if pw is None:
                pw = getpass.getpass(prompt=f"🔑 Enter password for {user}@{ip}: ")
                password_cache[user] = pw

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(ip, username=user, password=pw, timeout=5.0)

        # Ensure ~/.ssh exists
        client.exec_command("mkdir -p ~/.ssh && chmod 700 ~/.ssh")

        # Use SFTP to copy the key files
        sftp: SFTPClient = client.open_sftp()
        # NOTE: Paramiko SFTP does not expand '~', so we use relative paths
        remote_priv = f".ssh/{config.shared_key_name}"
        remote_pub = f".ssh/{config.shared_key_name}.pub"
        log.debug(f"{emoji} Uploading private key → {remote_priv}")
        sftp.put(config.key_path, remote_priv)
        log.debug(f"{emoji} Uploading public key → {remote_pub}")
        sftp.put(config.pub_key_path, remote_pub)
        sftp.close()

        # Append pub key to authorized_keys and set proper permissions
        chmod_cmd = (
            f"cat $HOME/{remote_pub} >> $HOME/.ssh/authorized_keys && "
            f"chmod 600 $HOME/.ssh/authorized_keys $HOME/{remote_priv}"
        )
        run_remote_command(client, chmod_cmd)

        client.close()
        log.debug(f"{emoji} Keys distributed and authorized_keys updated for {name} ({ip})")

def write_ssh_config(nodes, config: SetupNetConfig):
    log.info(f"🌐 Writing SSH config to {config.config_path}")
    with open(config.config_path, "w") as f:
        f.write("# Auto-generated SSH config\n\n")
        for node in nodes:
            emoji = node.get("emoji", "🌐")
            f.write(f"# {emoji} {node['name']}\n")
            f.write(f"Host {node['name']}\n")
            f.write(f"    HostName {node['ip']}\n")
            f.write(f"    User {node['user']}\n")
            f.write(f"    IdentityFile ~/.ssh/{config.shared_key_name}\n\n")
    os.chmod(config.config_path, 0o600)
    log.debug(f"🌐 SSH config written and permissions set.")

def test_node_connection(node, timeout: float) -> tuple[str, bool, str]:
    """Test SSH connectivity to a single node.
    
    Returns:
        tuple: (node_name, success, message)
    """
    if _is_local_node(node):
        return node["name"], True, f"🌐 {node['name']} is local: skipping connectivity test"
    name, ip, emoji = node["name"], node["ip"], node.get("emoji", "🌐")
    
    # First test if we can reach the host
    try:
        sock = socket.create_connection((ip, 22), timeout=timeout)
        sock.close()
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        return name, False, f"{emoji} {name} ({ip}): Connection failed - {str(e)}"
    
    # Then test SSH connectivity
    try:
        subprocess.run(
            f"ssh -o BatchMode=yes -o ConnectTimeout={int(timeout)} {name} echo 'Connection test'",
            shell=True, check=True, capture_output=True, text=True
        )
        return name, True, f"{emoji} {name} ({ip}): Connection successful"
    except subprocess.CalledProcessError as e:
        return name, False, f"{emoji} {name} ({ip}): SSH test failed - {e.stderr.strip()}"

def test_nodes(config: SetupNetConfig) -> bool:
    """Test connectivity to all nodes in parallel.
    
    Returns:
        bool: True if all nodes are reachable, False otherwise.
    """
    nodes = load_nodes(config.yaml_file)
    all_success = True
    
    log.info("🌐 Testing connectivity to all nodes...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(test_node_connection, node, config.test_timeout)
            for node in nodes
        ]
        
        for future in concurrent.futures.as_completed(futures):
            name, success, message = future.result()
            if success:
                log.info(message)
            else:
                log.error(message)
                all_success = False
    
    if all_success:
        log.info("🌐 ✅ All nodes are responding")
    else:
        log.error("🌐 ❌ Some nodes are not responding")
    
    return all_success

def setup_net(config: SetupNetConfig):
    nodes = load_nodes(config.yaml_file)
    generate_key(config=config)
    distribute_keys(nodes, config=config)
    write_ssh_config(nodes, config=config)
    log.info(f"🌐 SSH setup complete. Try: ssh node2")
    
    # Test connectivity to all nodes
    test_nodes(config)

def get_ssh_client(hostname: str, username: str, key_path: str) -> SSHClient:
    """Create and return a configured SSH client.
    
    Args:
        hostname: Remote host to connect to
        username: Username for SSH connection
        key_path: Path to private key file
    
    Returns:
        SSHClient: Connected SSH client
    
    Raises:
        paramiko.SSHException: If connection fails
        paramiko.AuthenticationException: If authentication fails
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(
            hostname=hostname,
            username=username,
            key_filename=key_path,
            timeout=5.0
        )
        log.debug(f"🌐 🔑 SSH connection established to {username}@{hostname}")
        return client
    except Exception as e:
        log.error(f"🌐 ❌ Failed to connect to {username}@{hostname}: {str(e)}")
        raise

def run_remote_command(
    client: SSHClient,
    command: str,
    timeout: float = 30.0
) -> Tuple[int, str, str]:
    """Run a command on the remote host.
    
    Args:
        client: Connected SSH client
        command: Command to execute
        timeout: Command timeout in seconds
        
    Returns:
        Tuple[int, str, str]: (exit_code, stdout, stderr)
    """
    log.info(f"🌐 🖥️ Running remote command: {command}")
    
    try:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        
        if exit_code != 0:
            log.error(f"🌐 ❌ Command failed with exit code {exit_code}")
            if err:
                log.error(f"🌐 stderr: {err}")
        else:
            log.debug(f"🌐 ✅ Command completed successfully")
            if out:
                log.debug(f"🌐 stdout: {out}")
                
        return exit_code, out, err
        
    except Exception as e:
        log.error(f"🌐 ❌ Failed to execute command: {str(e)}")
        raise

def transfer_file(
    client: SSHClient,
    local_path: str,
    remote_path: str,
    direction: str = "put"
) -> None:
    """Transfer a file to/from the remote host.
    
    Args:
        client: Connected SSH client
        local_path: Path to local file
        remote_path: Path on remote host
        direction: "put" to upload, "get" to download
        
    Raises:
        FileNotFoundError: If local file doesn't exist for upload
        paramiko.SFTPError: If transfer fails
    """
    if direction not in ["put", "get"]:
        raise ValueError("direction must be 'put' or 'get'")
        
    try:
        sftp: SFTPClient = client.open_sftp()
        
        if direction == "put":
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            log.info(f"🌐 📤 Uploading {local_path} to {remote_path}")
            sftp.put(local_path, remote_path)
            
        else:  # get
            log.info(f"🌐 📥 Downloading {remote_path} to {local_path}")
            sftp.get(remote_path, local_path)
            
        sftp.close()
        log.debug(f"🌐 ✅ File transfer completed successfully")
        
    except Exception as e:
        log.error(f"🌐 ❌ File transfer failed: {str(e)}")
        raise

if __name__ == "__main__":
    args = setup_log_with_config(SetupNetConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    setup_net(args)