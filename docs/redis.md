# Tatbot State Server (Redis)

This doc covers the Redis-based parameter server: architecture, setup on eek, usage, and troubleshooting.

## Architecture
- Redis runs on eek (192.168.1.97), LAN-only, no password.
- Clients use `StateManager` (src/tatbot/state) for snapshots, events, and streams.
- Viz can subscribe to stroke progress and update UI live.

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
- Tools: get_state, set_state, publish_event, subscribe_events, system_status
- Resources: state://status, state://stroke/progress, state://health/{node_id}

## Viz integration
- Start stroke viz and enable "Sync with Robot" in the GUI, or pass enable_state_sync=true via the viz MCP tool.

## Troubleshooting
- Connection refused: redis running? `redis-cli -h eek -p 6379 ping`; network OK? `ping eek`.
- No updates: check process; subscribe via `redis-cli psubscribe "stroke:events:*"`.

## Security & Performance
- LAN-restricted bind (127.0.0.1 and 192.168.1.97), protected-mode yes, no password.
- AOF everysec; light CPU/memory footprint for typical loads.

