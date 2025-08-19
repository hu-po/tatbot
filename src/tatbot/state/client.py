"""Redis client wrapper with connection management and retries."""

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from tatbot.utils.exceptions import NetworkConnectionError
from tatbot.utils.log import get_logger

log = get_logger("state.client", "🔄")


class RedisClient:
    """Async Redis client with connection pooling and error handling."""
    
    def __init__(
        self,
        host: str = "eek",
        port: int = 6379, 
        password: Optional[str] = None,
        db: int = 0,
        max_connections: int = 20,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        # Allow environment overrides
        self.host = os.environ.get("REDIS_HOST", host)
        try:
            self.port = int(os.environ.get("REDIS_PORT", str(port)))
        except ValueError:
            self.port = port
        self.password = password or os.environ.get("REDIS_PASSWORD", None)
        self.db = db
        self.max_connections = max_connections
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Redis server."""
        if self._connected and self._redis:
            return
            
        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                max_connections=self.max_connections,
                decode_responses=True,
                health_check_interval=30,
            )
            
            # Create Redis client
            self._redis = aioredis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            log.info(f"✅ Connected to Redis at {self.host}:{self.port}")
            
        except Exception as e:
            self._connected = False
            log.error(f"❌ Failed to connect to Redis: {e}")
            raise NetworkConnectionError(f"Cannot connect to Redis server at {self.host}:{self.port}: {e}")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
        
        if self._pool:
            await self._pool.aclose()
            self._pool = None
            
        self._connected = False
        log.info("Disconnected from Redis")

    async def _execute_with_retry(self, operation, *args, **kwargs) -> Any:
        """Execute Redis operation with retry logic."""
        if not self._connected or not self._redis:
            await self.connect()
        
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                log.warning(f"Redis operation failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                    # Try to reconnect
                    try:
                        await self.disconnect()
                        await self.connect()
                    except Exception as reconnect_error:
                        log.error(f"Reconnection failed: {reconnect_error}")
        
        raise NetworkConnectionError(f"Redis operation failed after {self.retry_attempts} attempts: {last_exception}")

    # Hash operations
    async def hset(self, key: str, field: str, value: Any) -> bool:
        """Set field in hash."""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        return await self._execute_with_retry(self._redis.hset, key, field, value)

    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get field from hash."""
        result = await self._execute_with_retry(self._redis.hget, key, field)
        return result

    async def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields from hash."""
        return await self._execute_with_retry(self._redis.hgetall, key)

    async def hdel(self, key: str, *fields: str) -> int:
        """Delete fields from hash."""
        return await self._execute_with_retry(self._redis.hdel, key, *fields)

    # String operations
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set key-value pair with optional expiration."""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        return await self._execute_with_retry(self._redis.set, key, value, ex=ex)

    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        return await self._execute_with_retry(self._redis.get, key)

    async def delete(self, *keys: str) -> int:
        """Delete keys."""
        return await self._execute_with_retry(self._redis.delete, *keys)

    # List operations  
    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to left of list."""
        serialized = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in values]
        return await self._execute_with_retry(self._redis.lpush, key, *serialized)

    async def rpop(self, key: str) -> Optional[str]:
        """Pop value from right of list."""
        return await self._execute_with_retry(self._redis.rpop, key)

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get range of values from list."""
        return await self._execute_with_retry(self._redis.lrange, key, start, end)

    async def llen(self, key: str) -> int:
        """Get length of list."""
        return await self._execute_with_retry(self._redis.llen, key)


    # Pub/Sub operations
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        if isinstance(message, (dict, list)):
            message = json.dumps(message)
        return await self._execute_with_retry(self._redis.publish, channel, message)

    async def subscribe(self, *channels: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to channels and yield messages."""
        if not self._connected or not self._redis:
            await self.connect()
            
        pubsub = self._redis.pubsub()
        try:
            await pubsub.subscribe(*channels)
            log.info(f"📡 Subscribed to channels: {channels}")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Try to parse as JSON
                        data = json.loads(message["data"])
                    except (json.JSONDecodeError, TypeError):
                        # Fallback to string
                        data = message["data"]
                    
                    yield {
                        "channel": message["channel"],
                        "data": data,
                        "pattern": message.get("pattern"),
                    }
        except Exception as e:
            log.error(f"Subscription error: {e}")
            raise
        finally:
            await pubsub.aclose()

    # Stream operations (for time-series data)
    async def xadd(self, stream: str, fields: Dict[str, Any], maxlen: Optional[int] = None) -> str:
        """Add entry to stream."""
        serialized_fields = {}
        for k, v in fields.items():
            if isinstance(v, (dict, list)):
                serialized_fields[k] = json.dumps(v)
            else:
                serialized_fields[k] = str(v)
        
        return await self._execute_with_retry(
            self._redis.xadd, stream, serialized_fields, maxlen=maxlen
        )

    async def xrange(self, stream: str, min_id: str = "-", max_id: str = "+", count: Optional[int] = None) -> List:
        """Read range from stream."""
        return await self._execute_with_retry(
            self._redis.xrange, stream, min_id, max_id, count=count
        )

    async def xlen(self, stream: str) -> int:
        """Get stream length."""
        return await self._execute_with_retry(self._redis.xlen, stream)

    # Utility operations
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        return await self._execute_with_retry(self._redis.exists, *keys)

    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration."""
        return await self._execute_with_retry(self._redis.expire, key, seconds)

    async def ttl(self, key: str) -> int:
        """Get key time-to-live."""
        return await self._execute_with_retry(self._redis.ttl, key)

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        return await self._execute_with_retry(self._redis.keys, pattern)

    async def ping(self) -> bool:
        """Ping Redis server."""
        try:
            result = await self._execute_with_retry(self._redis.ping)
            return result
        except Exception:
            return False

    # Context manager support
    async def __aenter__(self) -> "RedisClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()
