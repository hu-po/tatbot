from __future__ import annotations

import os
import socket
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import paramiko


@dataclass
class SSHConfig:
    key_path: Optional[str] = None
    port: Optional[int] = None
    timeout: float = 5.0


class SSHPool:
    def __init__(self, config: Optional[SSHConfig] = None):
        self._clients: dict[tuple[str, str, int], paramiko.SSHClient] = {}
        self._lock = threading.Lock()
        if config is None:
            key_env = os.environ.get("TATBOT_TUI_SSH_KEY")
            port_env = os.environ.get("TATBOT_TUI_SSH_PORT")
            config = SSHConfig(
                key_path=key_env,
                port=int(port_env) if port_env else None,
                timeout=5.0,
            )
        self.config = config

    def _client_key(self, host: str, user: str, port: int) -> tuple[str, str, int]:
        return (host, user, port)

    def _connect(self, host: str, user: str, port: int) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=host,
                port=port,
                username=user,
                key_filename=self.config.key_path,
                timeout=self.config.timeout,
                allow_agent=True,
                look_for_keys=True,
                banner_timeout=3,
                auth_timeout=4,
            )
        except Exception:
            # Ensure client is closed on failure
            try:
                client.close()
            except Exception:
                pass
            raise
        return client

    def _get_client(self, host: str, user: str) -> Tuple[paramiko.SSHClient, int]:
        port = self.config.port or 22
        key = self._client_key(host, user, port)
        with self._lock:
            client = self._clients.get(key)
            if client is not None:
                return client, port
            client = self._connect(host, user, port)
            self._clients[key] = client
            return client, port

    def exec(self, host: str, user: str, cmd: str, timeout: float = 4.0) -> tuple[int, str, str]:
        client, _ = self._get_client(host, user)
        try:
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                # force reconnect
                with self._lock:
                    # remove stale
                    for k, c in list(self._clients.items()):
                        if c is client:
                            del self._clients[k]
                            break
                client, _ = self._get_client(host, user)
            stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
            exit_status = stdout.channel.recv_exit_status()
            out = stdout.read().decode("utf-8", errors="ignore")
            err = stderr.read().decode("utf-8", errors="ignore")
            return exit_status, out, err
        except (socket.timeout, paramiko.SSHException) as e:
            return 124, "", str(e)
        except Exception as e:
            return 1, "", str(e)

    def close_all(self) -> None:
        with self._lock:
            for client in self._clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()


# Singleton used by callers
_pool: Optional[SSHPool] = None


def get_pool() -> SSHPool:
    global _pool
    if _pool is None:
        _pool = SSHPool()
    return _pool


def configure_pool(key_path: Optional[str] = None, port: Optional[int] = None) -> None:
    global _pool
    cfg = SSHConfig(key_path=key_path, port=port)
    _pool = SSHPool(cfg)
