#!/bin/bash
# Setup Redis server on eek node for tatbot parameter server

set -e

# Check if running on eek node
CURRENT_NODE=$(hostname)
if [ "$CURRENT_NODE" != "eek" ]; then
    echo "❌ This script must be run on the eek node, not $CURRENT_NODE"
    exit 1
fi

echo "🔄 Setting up Redis server on eek node..."

# Install Redis if not already installed
if ! command -v redis-server &> /dev/null; then
    echo "📦 Installing Redis server..."
    sudo apt-get update
    sudo apt-get install -y redis-server
else
    echo "✅ Redis server already installed"
fi

# Create Redis configuration directory
sudo mkdir -p /etc/redis

# Create custom Redis configuration
echo "📝 Creating Redis configuration..."
sudo tee /etc/redis/tatbot-redis.conf > /dev/null <<EOF
# Tatbot Redis Parameter Server Configuration
# Based on Redis 7.0+ defaults with modifications for distributed robotics

# Network
bind 0.0.0.0
port 6379
protected-mode yes
tcp-backlog 511
timeout 0
tcp-keepalive 300

# General
daemonize yes
supervised systemd
pidfile /var/run/redis/tatbot-redis.pid
loglevel notice
logfile /var/log/redis/tatbot-redis.log
databases 16

# Persistence - Enable both AOF and RDB for maximum durability
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename tatbot-dump.rdb
dir /var/lib/redis

# AOF (Append Only File) for durability
appendonly yes
appendfilename "tatbot-appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Lazy freeing
lazyfree-lazy-eviction no
lazyfree-lazy-expire no
lazyfree-lazy-server-del no
replica-lazy-flush no

# Security
requirepass CHANGE_ME_REDIS_PASSWORD

# Clients
maxclients 10000

# Networking optimizations for robotics
tcp-no-delay yes
repl-disable-tcp-nodelay no

# Pub/Sub optimizations
notify-keyspace-events Ex

# Disable dangerous commands in production
# rename-command FLUSHALL ""
# rename-command FLUSHDB ""
# rename-command DEBUG ""
# rename-command CONFIG ""
EOF

# Create systemd service file
echo "🛠️ Creating systemd service..."
sudo tee /etc/systemd/system/tatbot-redis.service > /dev/null <<EOF
[Unit]
Description=Tatbot Redis Parameter Server
After=network.target
Documentation=http://redis.io/documentation, man:redis-server(1)

[Service]
Type=notify
ExecStart=/usr/bin/redis-server /etc/redis/tatbot-redis.conf
ExecStop=/bin/redis-cli -p 6379 shutdown
TimeoutStopSec=0
Restart=always
User=redis
Group=redis
RuntimeDirectory=redis
RuntimeDirectoryMode=0755

# Security settings
NoNewPrivileges=true
PrivateTmp=yes
PrivateDevices=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/redis
ReadWritePaths=/var/log/redis
ReadWritePaths=/var/run/redis

[Install]
WantedBy=multi-user.target
EOF

# Create Redis directories with proper permissions
echo "📁 Creating Redis directories..."
sudo mkdir -p /var/lib/redis
sudo mkdir -p /var/log/redis
sudo mkdir -p /var/run/redis
sudo chown redis:redis /var/lib/redis /var/log/redis /var/run/redis
sudo chmod 755 /var/lib/redis /var/log/redis /var/run/redis

# Generate random password if not set
if [ ! -f ~/.redis_password ]; then
    echo "🔐 Generating Redis password..."
    REDIS_PASSWORD=$(openssl rand -base64 32)
    echo "REDIS_PASSWORD=$REDIS_PASSWORD" > ~/.redis_password
    chmod 600 ~/.redis_password
    
    # Update config with actual password
    sudo sed -i "s/CHANGE_ME_REDIS_PASSWORD/$REDIS_PASSWORD/" /etc/redis/tatbot-redis.conf
    
    echo "✅ Redis password saved to ~/.redis_password"
    echo "💡 Add this to your .env file: export REDIS_PASSWORD=$REDIS_PASSWORD"
else
    echo "✅ Using existing Redis password from ~/.redis_password"
    REDIS_PASSWORD=$(grep REDIS_PASSWORD ~/.redis_password | cut -d'=' -f2)
    sudo sed -i "s/CHANGE_ME_REDIS_PASSWORD/$REDIS_PASSWORD/" /etc/redis/tatbot-redis.conf
fi

# Reload systemd and enable service
echo "🔄 Enabling and starting Redis service..."
sudo systemctl daemon-reload
sudo systemctl enable tatbot-redis.service
sudo systemctl start tatbot-redis.service

# Wait for Redis to start
sleep 2

# Test Redis connection
echo "🧪 Testing Redis connection..."
if redis-cli -p 6379 -a "$REDIS_PASSWORD" ping | grep -q PONG; then
    echo "✅ Redis server is running and responding to pings"
else
    echo "❌ Redis server is not responding"
    sudo systemctl status tatbot-redis.service
    exit 1
fi

# Show service status
echo "📊 Redis service status:"
sudo systemctl status tatbot-redis.service --no-pager -l

echo ""
echo "🎉 Redis setup complete!"
echo ""
echo "Service management commands:"
echo "  sudo systemctl start tatbot-redis"
echo "  sudo systemctl stop tatbot-redis" 
echo "  sudo systemctl restart tatbot-redis"
echo "  sudo systemctl status tatbot-redis"
echo ""
echo "Redis CLI connection:"
echo "  redis-cli -p 6379 -a \$REDIS_PASSWORD"
echo ""
echo "Logs:"
echo "  sudo journalctl -u tatbot-redis -f"
echo "  tail -f /var/log/redis/tatbot-redis.log"
echo ""
echo "🔐 Remember to add REDIS_PASSWORD to your .env file on all nodes!"
echo "   export REDIS_PASSWORD=$REDIS_PASSWORD"