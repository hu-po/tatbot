#!/bin/bash
# Fix Redis setup issues on eek node

set -e

echo "🔧 Fixing Redis configuration on eek node..."

# Stop and disable default Redis service
echo "🛑 Stopping default Redis service..."
sudo systemctl stop redis-server || echo "Default Redis not running"
sudo systemctl disable redis-server || echo "Default Redis not enabled"

# Check if port 6379 is free
if sudo netstat -tulpn | grep :6379; then
    echo "❌ Port 6379 is still in use. Checking processes..."
    sudo lsof -i :6379 || true
    sudo pkill -f redis-server || true
    sleep 2
fi

# Test our configuration manually
echo "🧪 Testing tatbot Redis configuration..."
sudo /usr/bin/redis-server /etc/redis/tatbot-redis.conf --daemonize no --loglevel verbose &
REDIS_PID=$!

# Wait a moment then check if it's running
sleep 3
if kill -0 $REDIS_PID 2>/dev/null; then
    echo "✅ Configuration test successful"
    kill $REDIS_PID
    wait $REDIS_PID 2>/dev/null || true
else
    echo "❌ Configuration test failed"
    exit 1
fi

# Start our systemd service
echo "🚀 Starting tatbot-redis service..."
sudo systemctl daemon-reload
sudo systemctl start tatbot-redis
sleep 2

# Check service status
if sudo systemctl is-active --quiet tatbot-redis; then
    echo "✅ tatbot-redis service is running"
    sudo systemctl status tatbot-redis --no-pager -l
else
    echo "❌ tatbot-redis service failed to start"
    sudo systemctl status tatbot-redis --no-pager -l
    echo "--- Checking logs ---"
    sudo journalctl -u tatbot-redis --no-pager -l
    exit 1
fi

# Test Redis connection
REDIS_PASSWORD=$(grep REDIS_PASSWORD ~/.redis_password | cut -d'=' -f2)
echo "🧪 Testing Redis connection..."
if redis-cli -p 6379 -a "$REDIS_PASSWORD" ping | grep -q PONG; then
    echo "✅ Redis connection test successful"
    
    # Show connection info
    echo "📊 Redis info:"
    redis-cli -p 6379 -a "$REDIS_PASSWORD" info server | head -5
else
    echo "❌ Redis connection test failed"
    exit 1
fi

echo ""
echo "🎉 Redis setup complete!"
echo "✅ Service: tatbot-redis.service"
echo "✅ Port: 6379"
echo "✅ Password: $(echo $REDIS_PASSWORD | cut -c1-8)..."
echo ""
echo "Add to .env on all nodes:"
echo "export REDIS_PASSWORD=$REDIS_PASSWORD"