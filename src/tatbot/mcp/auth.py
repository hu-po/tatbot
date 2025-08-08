"""JWT token generation and validation for tatbot MCP servers."""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import jwt
from fastmcp.auth import BearerAuthProvider
from pydantic import BaseModel

from tatbot.utils.log import get_logger

log = get_logger("mcp.auth", "🔐")


class AuthConstants:
    """Authentication configuration constants."""
    DEFAULT_SECRET_KEY = "tatbot-mcp-secret-change-me-in-production"
    DEFAULT_ALGORITHM = "HS256"
    DEFAULT_EXPIRY_DAYS = 365
    TOKEN_ENV_VAR = "TATBOT_JWT_TOKEN"
    SECRET_ENV_VAR = "TATBOT_JWT_SECRET"


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    sub: str  # subject (node name)
    iat: int  # issued at
    exp: int  # expires at
    iss: str = "tatbot"  # issuer


class TatbotJWTAuth:
    """JWT token generator and validator for tatbot MCP servers."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize JWT auth with secret key."""
        self.secret_key = secret_key or os.getenv(
            AuthConstants.SECRET_ENV_VAR, 
            AuthConstants.DEFAULT_SECRET_KEY
        )
        self.algorithm = AuthConstants.DEFAULT_ALGORITHM
        
        if self.secret_key == AuthConstants.DEFAULT_SECRET_KEY:
            log.warning("Using default JWT secret key - change in production!")
    
    def generate_token(self, node_name: str, expiry_days: int = AuthConstants.DEFAULT_EXPIRY_DAYS) -> str:
        """Generate a JWT token for a specific node."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=expiry_days)
        
        payload = TokenPayload(
            sub=node_name,
            iat=int(now.timestamp()),
            exp=int(expiry.timestamp())
        )
        
        token = jwt.encode(
            payload.model_dump(),
            self.secret_key,
            algorithm=self.algorithm
        )
        
        log.info(f"Generated token for node '{node_name}' (expires: {expiry.isoformat()})")
        return token
    
    def validate_token(self, token: str) -> Optional[TokenPayload]:
        """Validate a JWT token and return payload if valid."""
        try:
            decoded = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return TokenPayload(**decoded)
        except jwt.ExpiredSignatureError:
            log.error("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            log.error(f"Invalid token: {e}")
            return None
    
    def create_bearer_auth_provider(self) -> BearerAuthProvider:
        """Create FastMCP BearerAuthProvider with validation function."""
        def verify_token(token: str) -> bool:
            """Token verification function for FastMCP."""
            payload = self.validate_token(token)
            if payload:
                log.info(f"Authenticated request from node: {payload.sub}")
                return True
            return False
        
        return BearerAuthProvider(verify_token)


def generate_node_tokens(nodes: Optional[list] = None) -> Dict[str, str]:
    """Generate tokens for all tatbot nodes."""
    if nodes is None:
        nodes = ["ook", "oop", "trossen-ai", "rpi1", "rpi2"]
    
    auth = TatbotJWTAuth()
    tokens = {}
    
    log.info(f"Generating tokens for nodes: {nodes}")
    
    for node in nodes:
        tokens[node] = auth.generate_token(node)
    
    return tokens


def print_env_vars(tokens: Dict[str, str]) -> None:
    """Print environment variable exports for tokens."""
    print("\n# Add these to your .env file or export in shell:")
    for node, token in tokens.items():
        print(f"export TATBOT_JWT_TOKEN_{node.upper().replace('-', '_')}={token}")
    
    print("\n# Shared secret (same for all nodes):")
    auth = TatbotJWTAuth()
    print(f"export TATBOT_JWT_SECRET={auth.secret_key}")


if __name__ == "__main__":
    """CLI for generating tokens."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate JWT tokens for tatbot MCP servers")
    parser.add_argument("--nodes", nargs="+", help="Node names to generate tokens for")
    parser.add_argument("--expiry-days", type=int, default=365, help="Token expiry in days")
    
    args = parser.parse_args()
    
    tokens = generate_node_tokens(args.nodes)
    
    print("Generated tokens:")
    for node, token in tokens.items():
        print(f"{node}: {token[:20]}...")
    
    print_env_vars(tokens)