# Tatbot State Management System

The tatbot system now includes a comprehensive Redis-based parameter server for sharing global state across all nodes. This enables real-time synchronization between robot operations, visualization tools, and monitoring systems.

## Architecture

The state management system consists of:

- **Redis Server**: Centralized parameter server running on the `eek` node
- **StateManager**: Python singleton class for state operations
- **MCP Integration**: State management tools accessible via MCP
- **Visualization Sync**: Real-time updates in visualization tools
- **Pub/Sub Events**: Event-driven communication between nodes

## Quick Start

### 1. Set Up Redis Server (eek node only)

```bash
# SSH to eek node
ssh eek

# Run setup script
cd ~/tatbot
./scripts/setup_redis.sh

# Copy the generated password to your .env file
cat ~/.redis_password
```

### 2. Configure All Nodes

Add the Redis password to `.env` on each node:

```bash
# Add to .env file on all nodes
echo "REDIS_PASSWORD=your_password_here" >> .env
```

### 3. Test the System

```bash
# Test from any node
cd ~/tatbot
./scripts/test_state_server.py
```

## State Types

### Robot State
- Connection status (left/right arms)
- Joint positions and velocities  
- Current pose (ready, sleep, executing)
- Goal time settings

### Stroke Progress
- Current stroke and pose indices
- Total strokes and stroke length
- Stroke descriptions for both arms
- Execution status (active, paused)
- Scene and dataset information

### Node Health
- System metrics (CPU, memory, disk, GPU)
- Network connectivity status
- MCP server status and ports
- Heartbeat timestamps

### Sensor Data
- Camera capture metadata
- Image and depth data paths
- Calibration status
- AprilTag detections

## Usage Examples

### Basic State Operations

```python
from tatbot.state import StateManager

# Initialize (singleton pattern)
state = StateManager(node_id="hog")

# Use as context manager
async with state:
    # Update robot state
    robot_state = RobotState(
        node_id="hog",
        is_connected_l=True,
        is_connected_r=True,
        current_pose="ready"
    )
    await state.update_robot_state(robot_state)
    
    # Get stroke progress
    progress = await state.get_stroke_progress()
    if progress:
        print(f"Stroke {progress.stroke_idx}/{progress.total_strokes}")
```

### Stroke Session Management

```python
# Start stroke execution session
session_id = await state.start_stroke_session(
    total_strokes=100,
    stroke_length=50,
    scene_name="tatbotlogo"
)

# Update progress during execution
await state.update_stroke_progress(
    stroke_idx=5,
    pose_idx=25,
    stroke_description_l="tattoo_line_left",
    stroke_description_r="ink_dip_right",
    session_id=session_id
)

# End session
await state.end_stroke_session(session_id)
```

### Event Subscription

```python
# Subscribe to stroke events
async for message in state.subscribe_events("stroke:events:progress"):
    data = message["data"]
    print(f"Stroke progress: {data['stroke_idx']}/{data['total_strokes']}")
```

## MCP Tools

The state management system provides MCP tools for remote access:

### Available Tools

- **`get_state`**: Retrieve state data by key
- **`set_state`**: Set state data with optional TTL
- **`publish_event`**: Publish events to channels
- **`subscribe_events`**: Subscribe to event channels
- **`system_status`**: Get overall system status

### MCP Resources

- **`state://status`**: Global system status
- **`state://stroke/progress`**: Current stroke progress
- **`state://health/{node_id}`**: Node health status

### Usage via MCP

```bash
# Get system status
curl -X POST http://eek:8080/mcp/ \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "system_status", "arguments": {}}}'

# Get stroke progress  
curl -X POST http://hog:8080/mcp/ \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "get_state", "arguments": {"key": "stroke:progress"}}}'
```

## Visualization Integration

The stroke visualization now includes real-time synchronization:

### Features

- **State Sync Checkbox**: Enable/disable real-time updates
- **Connection Status**: Shows Redis connection state
- **Auto-Update**: Visualization updates automatically during robot execution
- **Session Tracking**: Shows active stroke sessions

### Usage

1. Start stroke visualization: `uv run python -m tatbot.viz.stroke --scene=tatbotlogo`
2. Enable "Sync with Robot" checkbox in the GUI
3. Run stroke execution on robot node
4. Visualization updates automatically in real-time

## Redis Configuration

### Server Settings (eek node)

- **Host**: 0.0.0.0 (accessible from all nodes)
- **Port**: 6379 (Redis default)
- **Authentication**: Password protected
- **Persistence**: Both RDB and AOF enabled
- **Memory Limit**: 2GB with LRU eviction

### Performance Optimizations

- **Connection Pooling**: Up to 20 connections per client
- **Pub/Sub Notifications**: Keyspace events enabled
- **TCP Optimizations**: No-delay enabled for low latency
- **Retry Logic**: 3 attempts with exponential backoff

## Monitoring and Debugging

### Service Management

```bash
# On eek node
sudo systemctl status tatbot-redis
sudo systemctl restart tatbot-redis
sudo journalctl -u tatbot-redis -f
```

### Redis CLI Access

```bash
# Connect to Redis
redis-cli -h eek -p 6379 -a $REDIS_PASSWORD

# Check connected clients
redis-cli -h eek -p 6379 -a $REDIS_PASSWORD info clients

# Monitor commands in real-time
redis-cli -h eek -p 6379 -a $REDIS_PASSWORD monitor
```

### Key Inspection

```bash
# List all state keys
redis-cli -h eek -p 6379 -a $REDIS_PASSWORD keys "*"

# Check stroke progress
redis-cli -h eek -p 6379 -a $REDIS_PASSWORD hgetall stroke:progress

# Monitor stroke events
redis-cli -h eek -p 6379 -a $REDIS_PASSWORD psubscribe "stroke:events:*"
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if Redis is running: `sudo systemctl status tatbot-redis`
   - Verify network connectivity: `ping eek`
   - Check firewall settings

2. **Authentication Failed**  
   - Verify REDIS_PASSWORD in .env matches server config
   - Check password file: `cat ~/.redis_password`

3. **Slow Performance**
   - Monitor Redis memory: `redis-cli info memory`
   - Check network latency: `ping eek`
   - Review connection pool settings

4. **State Not Updating**
   - Check pub/sub subscriptions: `redis-cli pubsub channels`
   - Verify event publishing: `redis-cli monitor`
   - Look for error logs in MCP server output

### Log Locations

- **Redis Server**: `/var/log/redis/tatbot-redis.log`
- **MCP Server**: Check MCP server output
- **StateManager**: Part of tatbot logging system

## Security Considerations

- **Password Authentication**: All connections require authentication
- **Network Security**: Redis bound to private network only
- **Service Isolation**: Redis runs as dedicated user with limited permissions
- **Data Persistence**: Encrypted storage recommended for sensitive deployments

## Performance Characteristics

- **Latency**: Sub-millisecond operations for local network
- **Throughput**: >10,000 operations/second typical
- **Memory Usage**: ~100MB base + data storage
- **Network Traffic**: Minimal overhead for typical robotics operations

## Future Enhancements

- **Multi-Robot Coordination**: State sharing between multiple tatbot systems
- **Historical Data**: Time-series storage for performance analysis  
- **Web Dashboard**: Browser-based monitoring interface
- **Alert System**: Notifications for critical state changes
- **Backup/Restore**: Automated state backup and recovery