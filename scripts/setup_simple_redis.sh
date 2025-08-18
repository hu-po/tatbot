#!/bin/bash
# Simple Redis setup using default service with password

set -e

echo "🚀 Setting up Redis with default service..."

# Stop our custom service if running
sudo systemctl stop tatbot-redis || true
sudo systemctl disable tatbot-redis || true

# Get or create Redis password
if [ ! -f ~/.redis_password ]; then
    echo "🔐 Generating Redis password..."
    REDIS_PASSWORD=$(openssl rand -base64 32)
    echo "REDIS_PASSWORD=$REDIS_PASSWORD" > ~/.redis_password
    chmod 600 ~/.redis_password
else
    REDIS_PASSWORD=$(grep REDIS_PASSWORD ~/.redis_password | cut -d'=' -f2)
fi

# Create simple Redis config
echo "📝 Creating Redis configuration..."
sudo tee /etc/redis/redis.conf > /dev/null <<EOF
# Basic Redis configuration for tatbot
bind 0.0.0.0 127.0.0.1
port 6379
protected-mode yes
requirepass $REDIS_PASSWORD
timeout 0
tcp-keepalive 300

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Persistence
save 900 1
save 300 10
save 60 10000
dbfilename dump.rdb
dir /var/lib/redis

# AOF
appendonly yes
appendfilename "appendonly.aof"

# Memory
maxmemory 1gb
maxmemory-policy allkeys-lru

# Pub/Sub notifications for tatbot
notify-keyspace-events Ex

# Clients
maxclients 1000
EOF

# Enable and start default Redis service
echo "🔄 Starting Redis service..."
sudo systemctl enable redis-server
sudo systemctl restart redis-server

# Wait for Redis to start
sleep 3

# Test connection
echo "🧪 Testing Redis connection..."
if redis-cli -a "$REDIS_PASSWORD" ping | grep -q PONG; then
    echo "✅ Redis connection successful!"
    
    # Show Redis info
    echo "📊 Redis server info:"
    redis-cli -a "$REDIS_PASSWORD" info server | grep -E "(redis_version|process_id|tcp_port)"
    
    echo ""
    echo "🎉 Redis setup complete!"
    echo "📋 Connection details:"
    echo "   Host: eek"
    echo "   Port: 6379"
    echo "   Password: $(echo $REDIS_PASSWORD | cut -c1-8)..."
    echo ""
    echo "💡 Add to .env on all nodes:"
    echo "   export REDIS_PASSWORD=$REDIS_PASSWORD"
    
else
    echo "❌ Redis connection failed"
    sudo systemctl status redis-server --no-pager
    exit 1
fi