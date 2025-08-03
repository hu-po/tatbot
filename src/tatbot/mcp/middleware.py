"""MCP server middleware for authentication and security."""

import ipaddress
from typing import Optional

from fastapi import HTTPException, Request

from tatbot.mcp.models import MCPSettings


class MCPSecurityMiddleware:
    """Security middleware for MCP server."""
    
    def __init__(self, settings: MCPSettings):
        self.settings = settings
    
    def verify_ip_allowlist(self, client_ip: str) -> bool:
        """Check if client IP is in allowlist."""
        if not self.settings.ip_allowlist:
            return True  # No restrictions if allowlist is empty
            
        try:
            client_addr = ipaddress.ip_address(client_ip)
            for allowed_ip in self.settings.ip_allowlist:
                if ipaddress.ip_address(allowed_ip) == client_addr:
                    return True
        except ValueError:
            # Invalid IP format
            return False
        
        return False
    
    def verify_auth_token(self, auth_header: Optional[str]) -> bool:
        """Verify authentication token."""
        if not self.settings.require_auth or not self.settings.auth_token:
            return True  # No auth required
            
        if not auth_header:
            return False
            
        # Simple Bearer token authentication
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            return token == self.settings.auth_token
            
        return False
    
    async def __call__(self, request: Request, call_next):
        """Process request through security middleware."""
        # Get client IP
        client_ip = request.client.host
        
        # Check IP allowlist
        if not self.verify_ip_allowlist(client_ip):
            raise HTTPException(
                status_code=403,
                detail=f"Client IP {client_ip} not in allowlist"
            )
        
        # Check authentication
        auth_header = request.headers.get("Authorization")
        if not self.verify_auth_token(auth_header):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        response = await call_next(request)
        return response