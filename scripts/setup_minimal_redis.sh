#!/bin/bash
# Minimal Redis setup - just add password to default config

set -e

echo "🚀 Setting up minimal Redis with password..."

# Stop any Redis services
sudo systemctl stop redis-server || true
sudo systemctl stop tatbot-redis || true

# Get or create password
if [ ! -f ~/.redis_password ]; then
    REDIS_PASSWORD=$(openssl rand -base64 32)
    echo "REDIS_PASSWORD=$REDIS_PASSWORD" > ~/.redis_password
    chmod 600 ~/.redis_password
else
    REDIS_PASSWORD=$(grep REDIS_PASSWORD ~/.redis_password | cut -d'=' -f2)
fi

# Backup original config
sudo cp /etc/redis/redis.conf /etc/redis/redis.conf.backup 2>/dev/null || true

# Modify default Redis config to add password and allow external connections
sudo sed -i 's/^bind 127.0.0.1 -::1/bind 0.0.0.0/' /etc/redis/redis.conf
sudo sed -i 's/^# requirepass foobared/requirepass '"$REDIS_PASSWORD"'/' /etc/redis/redis.conf

# Enable keyspace notifications for tatbot events
if ! grep -q "notify-keyspace-events" /etc/redis/redis.conf; then
    echo "notify-keyspace-events Ex" | sudo tee -a /etc/redis/redis.conf
fi

# Start Redis
echo "🔄 Starting Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Wait and test
sleep 3

echo "🧪 Testing connection..."
if redis-cli -a "$REDIS_PASSWORD" ping | grep -q PONG; then
    echo "✅ Redis is working!"
    echo "📊 Redis info:"
    redis-cli -a "$REDIS_PASSWORD" info server | head -3
    echo ""
    echo "🎉 Setup complete!"
    echo "💡 Add to all nodes' .env:"
    echo "export REDIS_PASSWORD=$REDIS_PASSWORD"
else
    echo "❌ Redis test failed"
    sudo systemctl status redis-server
fi