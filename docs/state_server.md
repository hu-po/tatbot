# Tatbot State Server (Redis)

This doc covers the Redis-based parameter server: architecture, setup on eek, usage, and troubleshooting.

## Architecture
- Redis runs on eek (192.168.1.97), LAN-only, no password.
- Clients use `StateManager` (src/tatbot/state) for snapshots, events, and streams.
- Viz can subscribe to stroke progress and update UI live.

## Network Modes (HOME vs EDGE)
- In EDGE mode, hosts like `eek` may be resolvable by name on the robot LAN.
- In HOME mode, hostnames might not resolve. Ensure components use the Redis IP address.
- Recommended:
  - Set `REDIS_HOST` and `REDIS_PORT` in `/nfs/tatbot/.env` (sourced by scripts) or in your shell before launching.
  - Example: `export REDIS_HOST=192.168.1.97; export REDIS_PORT=6379`
  - Optionally add `/etc/hosts` entries mapping `eek` to the Redis IP on each node.

## Design Notes: Redis vs MCP
- Redis is the system’s parameter/state server. Producers and consumers (robot, viz, services) should publish/subscribe and read/write state directly to Redis.
- MCP is for orchestration and read-only summaries (e.g., `state://status`), not real-time pub/sub during strokes.
- We do not expose Redis state tools over MCP anymore. Use direct Redis for live flows. MCP still provides read-only resources (e.g., `state://status`).

## Redis Configuration (Single Source of Truth)
- Redis host/port are configured via Hydra: `src/conf/redis/default.yaml`.
- The MCP server reads this config and also sets `REDIS_HOST`/`REDIS_PORT` environment variables in-process so any internal components using `StateManager()` pick the same target automatically.
- Adjust `host` or `port` in the Hydra file if your Redis location changes.

## Setup (eek node)
```bash
ssh eek
sudo mkdir -p /etc/redis
sudo cp ~/tatbot/config/redis/redis.conf /etc/redis/tatbot-redis.conf
sudo mkdir -p /var/run/redis /var/log/redis /var/lib/redis
sudo chown -R redis:redis /var/run/redis /var/log/redis /var/lib/redis 2>/dev/null || true
sudo redis-server /etc/redis/tatbot-redis.conf --daemonize yes
redis-cli -h eek -p 6379 ping
```

## Manage (no systemd)
```bash
# Stop
redis-cli -h eek -p 6379 shutdown 2>/dev/null || true
# Start
sudo redis-server /etc/redis/tatbot-redis.conf --daemonize yes
# Check
ps aux | grep redis-server | grep -v grep
```

## CLI examples
```bash
redis-cli -h eek -p 6379
redis-cli -h eek -p 6379 info clients
redis-cli -h eek -p 6379 monitor
redis-cli -h eek -p 6379 keys "*"
redis-cli -h eek -p 6379 psubscribe "stroke:events:*"
```

MCP exposes only read-only resources relevant to state. For example, see `state://status` below.

## Using StateManager
See src/tatbot/state/ for full APIs. Typical flow:
```python
from tatbot.state.manager import StateManager
state = StateManager(node_id="hog")
async with state:
    session = await state.start_stroke_session(total_strokes=100, stroke_length=50)
    await state.update_stroke_progress(stroke_idx=5, pose_idx=25, session_id=session)
    await state.end_stroke_session(session)
```

## MCP integration
- Tools: none (state tools removed; use Redis directly)
- Resources: state://status, state://stroke/progress, state://health/{node_id}

To launch the MCP server with the correct Redis target in HOME mode:
```bash
export REDIS_HOST=192.168.1.97  # your eek IP
./scripts/mcp_run.sh eek
```

## Viz integration
- Start stroke viz and enable "Sync with Robot" in the GUI, or pass enable_state_sync=true via the viz MCP tool.

## Troubleshooting
- Connection refused: redis running? `redis-cli -h eek -p 6379 ping`; network OK? `ping eek`.
- No updates: check process; subscribe via `redis-cli psubscribe "stroke:events:*"`.

MCP timeouts during strokes on robot nodes are expected; use direct Redis for pub/sub and state access.

## Security & Performance
- LAN-restricted bind (127.0.0.1 and 192.168.1.97), protected-mode yes, no password.
- AOF everysec; light CPU/memory footprint for typical loads.
