"""Integration tests for MCP server endpoints."""

import asyncio
import multiprocessing
import time

import httpx
import pytest
from omegaconf import OmegaConf

from tatbot.mcp.server import main as serve


class MCPTestServer:
    """Test server context manager."""
    
    def __init__(self, node_name: str = "ook", port: int = 9000):
        self.node_name = node_name
        self.port = port
        self.process = None
        
    def __enter__(self):
        # Create a minimal config for testing
        cfg = OmegaConf.create({
            "node": self.node_name,
            "mcp": {
                "host": "127.0.0.1",
                "port": self.port,
                "transport": "streamable-http",
                "debug": True,
                "extras": [],
                "tools": ["ping_nodes", "list_scenes", "list_nodes"]
            }
        })
        
        # Start server in a separate process
        self.process = multiprocessing.Process(
            target=lambda: serve(cfg)
        )
        self.process.start()
        
        # Wait for server to start
        time.sleep(2)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()


@pytest.mark.asyncio
async def test_ping_nodes_endpoint():
    """Test the ping_nodes endpoint."""
    with MCPTestServer() as server:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{server.port}/tools/ping_nodes",
                json={"nodes": None},
                timeout=10.0
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "details" in data
            assert "all_success" in data
            assert isinstance(data["details"], list)
            assert isinstance(data["all_success"], bool)


@pytest.mark.asyncio
async def test_list_scenes_endpoint():
    """Test the list_scenes endpoint."""
    with MCPTestServer() as server:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{server.port}/tools/list_scenes",
                json={},
                timeout=10.0
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "scenes" in data
            assert "count" in data
            assert isinstance(data["scenes"], list)
            assert isinstance(data["count"], int)
            assert data["count"] == len(data["scenes"])


@pytest.mark.asyncio
async def test_list_nodes_endpoint():
    """Test the list_nodes endpoint."""
    with MCPTestServer() as server:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{server.port}/tools/list_nodes",
                json={},
                timeout=10.0
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "nodes" in data
            assert "count" in data
            assert isinstance(data["nodes"], list)
            assert isinstance(data["count"], int)
            assert data["count"] == len(data["nodes"])


@pytest.mark.asyncio
async def test_invalid_tool_endpoint():
    """Test calling a non-existent tool."""
    with MCPTestServer() as server:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{server.port}/tools/invalid_tool",
                json={},
                timeout=10.0
            )
            
            # Should return 404 or appropriate error status
            assert response.status_code in [404, 422, 500]


@pytest.mark.asyncio 
async def test_ping_nodes_with_specific_nodes():
    """Test ping_nodes with specific node list."""
    with MCPTestServer() as server:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{server.port}/tools/ping_nodes",
                json={"nodes": ["ook"]},
                timeout=10.0
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "details" in data
            assert isinstance(data["details"], list)


@pytest.mark.asyncio
async def test_server_health():
    """Test basic server health/connectivity."""
    with MCPTestServer() as server:
        async with httpx.AsyncClient() as client:
            # Test if server is responding at all
            try:
                response = await client.get(
                    f"http://127.0.0.1:{server.port}/",
                    timeout=5.0
                )
                # Server should respond (even if it's an error response)
                assert response.status_code in [200, 404, 405]
            except httpx.ConnectError:
                pytest.fail("Server is not responding to HTTP requests")


if __name__ == "__main__":
    # Run a simple test if called directly
    asyncio.run(test_ping_nodes_endpoint())