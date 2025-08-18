#!/bin/bash
# Create a simple, working Redis configuration for tatbot

echo "📝 Creating simple Redis configuration..."

sudo tee /etc/redis/tatbot-redis.conf > /dev/null <<'EOF'
# Tatbot Redis Parameter Server - Simple Configuration
# Network
bind 0.0.0.0
port 6379
protected-mode yes
timeout 0
tcp-keepalive 300

# General
daemonize yes
supervised systemd
pidfile /var/run/redis/tatbot-redis.pid
loglevel notice
logfile /var/log/redis/tatbot-redis.log
databases 16

# Persistence
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename tatbot-dump.rdb
dir /var/lib/redis

# AOF
appendonly yes
appendfilename "tatbot-appendonly.aof"
appendfsync everysec

# Memory
maxmemory 1gb
maxmemory-policy allkeys-lru

# Security
requirepass REDIS_PASSWORD_PLACEHOLDER

# Clients
maxclients 1000

# Pub/Sub
notify-keyspace-events Ex
EOF

# Replace password placeholder
REDIS_PASSWORD=$(grep REDIS_PASSWORD ~/.redis_password | cut -d'=' -f2)
sudo sed -i "s/REDIS_PASSWORD_PLACEHOLDER/$REDIS_PASSWORD/" /etc/redis/tatbot-redis.conf

echo "✅ Simple Redis configuration created"