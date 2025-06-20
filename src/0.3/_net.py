import concurrent.futures
from dataclasses import dataclass
import getpass
import logging
import os
import socket
import subprocess
from typing import Optional, Tuple, List

import paramiko
from paramiko.client import SSHClient
from paramiko.sftp_client import SFTPClient
import yaml

from _log import get_logger, setup_log_with_config, print_config

log = get_logger('_net')

@dataclass
class Node:
    name: str
    """Name of the node (a computer within the tatbot network)."""
    ip: str
    """IP address of the node, used for SSH connection."""
    user: str
    """Username for SSH connection."""
    emoji: str = "🌐"
    """Emoji to use for logging."""
    deps: str = "."
    """Dependencies to install on the node, see pyproject.toml."""

@dataclass
class NetworkManagerConfig:
    debug: bool = False
    """Enable debug logging."""
    yaml_file: str = os.path.expanduser("~/tatbot/config/nodes.yaml")
    """Path to node config with all node information."""
    shared_key_name: str = "tatbot-key"
    """Name of the shared SSH key."""
    ssh_dir: str = os.path.expanduser("~/.ssh")
    """Path to the SSH directory."""
    key_path: str = os.path.join(ssh_dir, shared_key_name)
    """Path to the private key."""
    pub_key_path: str = f"{key_path}.pub"
    """Path to the public key."""
    ssh_config_path: str = os.path.join(ssh_dir, "config")
    """Path to the SSH config file."""
    test_timeout: float = 5.0
    """Timeout in seconds for node connectivity tests."""


class NetworkManager:
    """Manages network operations for all tatbot nodes."""

    def __init__(self, config: Optional[NetworkManagerConfig] = None):
        """
        Initializes the NetworkManager.
        Args:
            config: A NetworkManagerConfig instance. If None, a default is created.
        """
        if config is None:
            config = NetworkManagerConfig()
        self.config = config
        self.nodes: List[Node] = self._load_nodes()

        # Pre-compute local identifiers to avoid repeated system calls
        self._local_ips = self._get_local_ips()
        self._local_hostnames = {socket.gethostname(), socket.getfqdn(), "localhost"}

    def _load_nodes(self) -> List[Node]:
        """Loads node definitions from the YAML file specified in the config."""
        log.info(f"🌐 Loading nodes from {self.config.yaml_file}")
        try:
            with open(self.config.yaml_file, "r") as f:
                nodes_data = yaml.safe_load(f)["nodes"]
            nodes = [Node(**n) for n in nodes_data]
            log.debug(f"🌐 Loaded {len(nodes)} nodes: {[n.name for n in nodes]}")
            return nodes
        except FileNotFoundError:
            log.error(f"Node configuration file not found at {self.config.yaml_file}")
            return []
        except (yaml.YAMLError, KeyError, TypeError) as e:
            log.error(f"Error parsing node configuration file: {e}")
            return []

    def get_target_nodes(
        self, node_names: Optional[List[str]] = None
    ) -> Tuple[List[Node], Optional[str]]:
        """
        Filters nodes based on a list of names.

        Args:
            node_names: A list of node names to select. If None, all nodes are returned.

        Returns:
            A tuple containing the list of target node objects and an error string if any names were invalid.
        """
        if not node_names:
            return self.nodes, None

        valid_node_names = {n.name for n in self.nodes}
        invalid_nodes = set(node_names) - valid_node_names
        if invalid_nodes:
            return [], f"Error: Invalid node names provided: {', '.join(invalid_nodes)}"

        target_nodes = [n for n in self.nodes if n.name in node_names]
        return target_nodes, None

    def _get_local_ips(self) -> set[str]:
        """Returns a set of IP addresses assigned to the local host."""
        ips: set[str] = set()
        hostname = socket.gethostname()
        try:
            ips.update(socket.gethostbyname_ex(hostname)[2])
        except socket.gaierror:
            pass
        try:
            addrinfo = socket.getaddrinfo(hostname, None)
            ips.update(ai[4][0] for ai in addrinfo)
        except socket.gaierror:
            pass
        ips.update({"127.0.0.1", "0.0.0.0"})
        return ips

    def is_local_node(self, node: Node) -> bool:
        """Checks if a node object represents the local host."""
        return node.ip in self._local_ips or node.name in self._local_hostnames

    def get_ssh_client(self, hostname: str, username: str) -> SSHClient:
        """
        Creates and returns a configured SSH client.

        Args:
            hostname: Remote host to connect to.
            username: Username for the SSH connection.

        Returns:
            A connected paramiko.SSHClient instance.
        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=hostname,
                username=username,
                key_filename=self.config.key_path,
                timeout=5.0,
            )
            log.debug(f"🌐 🔑 SSH connection established to {username}@{hostname}")
            return client
        except Exception as e:
            log.error(f"🌐 ❌ Failed to connect to {username}@{hostname}: {str(e)}")
            raise

    def setup_network(self):
        """Runs the full network setup process."""
        self._generate_key()
        self._distribute_keys()
        self._write_ssh_config()
        log.info("🌐 SSH setup complete. Try: ssh <node_name>")
        self.test_all_nodes()

    def _run_local(self, command, **kwargs):
        log.info(f"🌐 Running: {command}")
        subprocess.run(command, shell=True, check=True, **kwargs)

    def _generate_key(self):
        """Generates the shared SSH key if it doesn't exist."""
        if not os.path.exists(self.config.key_path):
            log.info(f"🌐 Generating SSH key at {self.config.key_path}...")
            self._run_local(f"ssh-keygen -t rsa -b 4096 -f {self.config.key_path} -N ''")
        else:
            log.debug(f"🌐 SSH key already exists at {self.config.key_path}. Skipping generation.")

    def _distribute_keys(self):
        """Distributes the public key to all remote nodes."""
        password_cache: dict[str, str] = {}
        for node in self.nodes:
            if self.is_local_node(node):
                log.debug(f"🌐 🛑 Skipping key distribution for local node {node.name}")
                continue

            log.info(f"{node.emoji} Setting up {node.name} ({node.ip})")

            try:
                client = self.get_ssh_client(node.ip, node.user)
                log.debug(f"{node.emoji} Connected with existing key – no password needed 🎉")
            except Exception:
                pw = password_cache.get(node.user)
                if pw is None:
                    pw = getpass.getpass(prompt=f"🔑 Enter password for {node.user}@{node.ip}: ")
                    password_cache[node.user] = pw
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(node.ip, username=node.user, password=pw, timeout=5.0)

            # Ensure ~/.ssh exists and has correct permissions
            client.exec_command("mkdir -p ~/.ssh && chmod 700 ~/.ssh")

            # Upload keys via SFTP
            sftp: SFTPClient = client.open_sftp()
            remote_priv = f".ssh/{self.config.shared_key_name}"
            remote_pub = f".ssh/{self.config.shared_key_name}.pub"
            log.debug(f"{node.emoji} Uploading private key → {remote_priv}")
            sftp.put(self.config.key_path, remote_priv)
            log.debug(f"{node.emoji} Uploading public key → {remote_pub}")
            sftp.put(self.config.pub_key_path, remote_pub)
            sftp.close()

            # Add key to authorized_keys and set file permissions
            chmod_cmd = (
                f"cat $HOME/{remote_pub} >> $HOME/.ssh/authorized_keys && "
                f"chmod 600 $HOME/.ssh/authorized_keys $HOME/{remote_priv}"
            )
            self._run_remote_command(client, chmod_cmd)
            client.close()
            log.debug(f"{node.emoji} Keys distributed for {node.name}")

    def _write_ssh_config(self):
        """Writes the SSH config file for all nodes."""
        log.info(f"🌐 Writing SSH config to {self.config.ssh_config_path}")
        with open(self.config.ssh_config_path, "w") as f:
            f.write(f"# Auto-generated by tatbot at {self.config.ssh_config_path}\n\n")
            for node in self.nodes:
                f.write(f"# {node.emoji} {node.name}\n")
                f.write(f"Host {node.name}\n")
                f.write(f"    HostName {node.ip}\n")
                f.write(f"    User {node.user}\n")
                f.write(f"    IdentityFile {self.config.key_path}\n\n")
        os.chmod(self.config.ssh_config_path, 0o600)
        log.debug("🌐 SSH config written and permissions set.")

    def _run_remote_command(
        self, client: SSHClient, command: str, timeout: float = 30.0
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

    def _transfer_file(
        self, client: SSHClient, local_path: str, remote_path: str, direction: str = "put"
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

    def run_command_on_nodes(
        self, command: str, node_names: Optional[List[str]] = None, timeout: float = 30.0
    ) -> dict[str, Tuple[int, str, str]]:
        """
        Runs a command on specified nodes in parallel.

        Args:
            command: The command to execute on each remote node.
            node_names: A list of node names to run the command on. If None, runs on all nodes.
            timeout: Command timeout in seconds.

        Returns:
            A dictionary mapping node name to a tuple of (exit_code, stdout, stderr).
        """
        target_nodes, err = self.get_target_nodes(node_names)
        if err:
            log.error(err)
            return {}

        results = {}
        log.info(f"🌐 🖥️ Running command on {len(target_nodes)} nodes: {command}")

        def _run(node: Node):
            if self.is_local_node(node):
                log.debug(f"🌐 🛑 Skipping remote command for local node {node.name}")
                return node.name, (-1, "Local node, command not run", "")

            try:
                client = self.get_ssh_client(node.ip, node.user)
                result = self._run_remote_command(client, command, timeout)
                client.close()
                return node.name, result
            except Exception as e:
                log.error(f"🌐 ❌ Failed to run command on {node.name}: {e}")
                return node.name, (-1, "", str(e))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(_run, node): node for node in target_nodes}
            for future in concurrent.futures.as_completed(future_to_node):
                name, result = future.result()
                results[name] = result

        return results

    def transfer_files_to_nodes(
        self,
        local_path: str,
        remote_path: str,
        node_names: Optional[List[str]] = None,
        direction: str = "put",
    ):
        """
        Transfers a file to or from specified nodes in parallel.

        Args:
            local_path: Path to the local file.
            remote_path: Path on the remote host.
            node_names: A list of node names for the transfer. If None, targets all nodes.
            direction: "put" to upload, "get" to download.
        """
        target_nodes, err = self.get_target_nodes(node_names)
        if err:
            log.error(err)
            return

        log.info(f"🌐 🔄 Transferring files to {len(target_nodes)} nodes...")

        def _transfer(node: Node):
            if self.is_local_node(node):
                log.debug(f"🌐 🛑 Skipping file transfer for local node {node.name}")
                return

            try:
                client = self.get_ssh_client(node.ip, node.user)
                self._transfer_file(client, local_path, remote_path, direction)
                client.close()
            except Exception as e:
                log.error(f"🌐 ❌ Failed to transfer file for {node.name}: {e}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_transfer, node) for node in target_nodes]
            concurrent.futures.wait(futures)

    def test_all_nodes(self) -> Tuple[bool, list[str]]:
        """
        Tests connectivity to all configured nodes in parallel.

        Returns:
            A tuple containing a boolean (True if all nodes responded) and a list of status messages.
        """
        all_success = True
        messages: list[str] = []

        log.info("🌐 Testing connectivity to all nodes...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._test_node_connection, node): node for node in self.nodes
            }
            for future in concurrent.futures.as_completed(futures):
                _name, success, message = future.result()
                messages.append(message)
                if success:
                    log.info(message)
                else:
                    log.error(message)
                    all_success = False

        if all_success:
            log.info("🌐 ✅ All nodes are responding")
        else:
            log.error("🌐 ❌ Some nodes are not responding")

        return all_success, sorted(messages)

    def _test_node_connection(self, node: Node) -> tuple[str, bool, str]:
        """
        Tests SSH connectivity to a single node.

        Returns:
            A tuple: (node_name, success_boolean, message_string)
        """
        if self.is_local_node(node):
            return node.name, True, f"✅ {node.emoji} {node.name} (local) is reachable."

        try:
            sock = socket.create_connection((node.ip, 22), timeout=self.config.test_timeout)
            sock.close()
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            return node.name, False, f"❌ {node.emoji} {node.name} ({node.ip}): Unreachable - {e}"

        try:
            self.get_ssh_client(node.ip, node.user)
            return node.name, True, f"✅ {node.emoji} {node.name} ({node.ip}): SSH connection successful."
        except Exception as e:
            return node.name, False, f"❌ {node.emoji} {node.name} ({node.ip}): SSH connection failed - {e}"


if __name__ == "__main__":
    args = setup_log_with_config(NetworkManagerConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)

    manager = NetworkManager(args)
    manager.setup_network()