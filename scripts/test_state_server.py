#!/usr/bin/env python3
"""Test script for tatbot Redis parameter server functionality."""

import asyncio
import json
import time
from datetime import datetime

from tatbot.state.manager import StateManager
from tatbot.state.models import NodeHealth, RobotState
from tatbot.state.schemas import RedisKeySchema


async def test_basic_connection():
    """Test basic Redis connection."""
    print("🔌 Testing basic Redis connection...")
    
    try:
        state = StateManager(node_id="test_node")
        async with state:
            connected = await state.redis.ping()
            if connected:
                print("✅ Redis connection successful")
                return True
            else:
                print("❌ Redis ping failed")
                return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False


async def test_robot_state():
    """Test robot state management."""
    print("\n🤖 Testing robot state management...")
    
    try:
        state = StateManager(node_id="test_hog")
        async with state:
            # Create and update robot state
            robot_state = RobotState(
                node_id="test_hog",
                is_connected_l=True,
                is_connected_r=True,
                current_pose="ready",
            )
            
            await state.update_robot_state(robot_state)
            print("✅ Robot state updated")
            
            # Retrieve robot state
            retrieved_state = await state.get_robot_state("test_hog")
            if retrieved_state:
                print(f"✅ Robot state retrieved: {retrieved_state.current_pose}")
                return True
            else:
                print("❌ Robot state retrieval failed")
                return False
                
    except Exception as e:
        print(f"❌ Robot state test failed: {e}")
        return False


async def test_stroke_session():
    """Test stroke session management."""
    print("\n🖌️ Testing stroke session management...")
    
    try:
        state = StateManager(node_id="test_hog")
        async with state:
            # Start stroke session
            session_id = await state.start_stroke_session(
                total_strokes=10,
                stroke_length=50,
                scene_name="test_scene"
            )
            print(f"✅ Stroke session started: {session_id}")
            
            # Update progress multiple times
            for stroke_idx in range(3):
                for pose_idx in [0, 10, 25, 49]:
                    await state.update_stroke_progress(
                        stroke_idx=stroke_idx,
                        pose_idx=pose_idx,
                        stroke_description_l=f"test_stroke_l_{stroke_idx}",
                        stroke_description_r=f"test_stroke_r_{stroke_idx}",
                        session_id=session_id
                    )
                    await asyncio.sleep(0.1)  # Small delay
            
            print("✅ Stroke progress updated")
            
            # Get progress
            progress = await state.get_stroke_progress(session_id)
            if progress:
                print(f"✅ Progress retrieved: stroke {progress.stroke_idx}/{progress.total_strokes}")
            
            # End session
            await state.end_stroke_session(session_id)
            print("✅ Stroke session ended")
            
            return True
            
    except Exception as e:
        print(f"❌ Stroke session test failed: {e}")
        return False


async def test_pub_sub():
    """Test pub/sub messaging."""
    print("\n📡 Testing pub/sub messaging...")
    
    try:
        # Publisher
        publisher = StateManager(node_id="test_publisher")
        
        # Subscriber
        subscriber = StateManager(node_id="test_subscriber")
        
        messages_received = []
        
        async def subscriber_task():
            async with subscriber:
                count = 0
                async for message in subscriber.subscribe_events(RedisKeySchema.STROKE_EVENTS):
                    messages_received.append(message)
                    count += 1
                    if count >= 3:  # Receive 3 messages then exit
                        break
        
        async def publisher_task():
            async with publisher:
                # Wait a bit for subscriber to start
                await asyncio.sleep(1)
                
                # Publish test events
                for i in range(3):
                    await publisher.publish_event(
                        RedisKeySchema.STROKE_EVENTS,
                        {
                            "type": "test_event",
                            "message": f"Test message {i}",
                            "test_data": {"value": i * 10}
                        }
                    )
                    await asyncio.sleep(0.5)
        
        # Run subscriber and publisher concurrently
        await asyncio.gather(
            subscriber_task(),
            publisher_task()
        )
        
        if len(messages_received) >= 3:
            print(f"✅ Pub/sub test successful - received {len(messages_received)} messages")
            return True
        else:
            print(f"❌ Pub/sub test failed - only received {len(messages_received)} messages")
            return False
            
    except Exception as e:
        print(f"❌ Pub/sub test failed: {e}")
        return False


async def test_node_health():
    """Test node health management."""
    print("\n❤️ Testing node health management...")
    
    try:
        state = StateManager(node_id="test_node")
        async with state:
            # Update node health
            health = NodeHealth(
                node_id="test_node",
                cpu_percent=45.6,
                memory_percent=62.3,
                is_reachable=True,
                mcp_server_running=True,
                mcp_port=8080
            )
            
            await state.update_node_health(health)
            print("✅ Node health updated")
            
            # Retrieve node health
            retrieved_health = await state.get_node_health("test_node")
            if retrieved_health:
                print(f"✅ Node health retrieved: CPU {retrieved_health.cpu_percent}%")
            
            # Get all nodes health
            all_health = await state.get_all_nodes_health()
            print(f"✅ All nodes health retrieved: {len(all_health)} nodes")
            
            return True
            
    except Exception as e:
        print(f"❌ Node health test failed: {e}")
        return False


async def test_system_status():
    """Test system status overview."""
    print("\n📊 Testing system status...")
    
    try:
        state = StateManager(node_id="test_node")
        async with state:
            status = await state.get_system_status()
            
            print(f"✅ System status retrieved:")
            print(f"   Redis connected: {status['redis_connected']}")
            print(f"   Active sessions: {status['active_stroke_sessions']}")
            print(f"   Nodes online: {status['nodes_online']}/{status['total_nodes']}")
            
            return True
            
    except Exception as e:
        print(f"❌ System status test failed: {e}")
        return False


async def test_error_reporting():
    """Test error reporting."""
    print("\n⚠️ Testing error reporting...")
    
    try:
        state = StateManager(node_id="test_node")
        async with state:
            await state.report_error(
                "test_error",
                "This is a test error message",
                {"component": "test_script", "severity": "low"}
            )
            print("✅ Error reported successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error reporting test failed: {e}")
        return False


async def cleanup_test_data():
    """Clean up test data."""
    print("\n🧹 Cleaning up test data...")
    
    try:
        state = StateManager(node_id="cleanup")
        async with state:
            # Clean up test keys
            test_keys = [
                "robot:state:test_hog",
                "node:health:test_node",
                "node:health:test_publisher", 
                "node:health:test_subscriber",
            ]
            
            for key in test_keys:
                if await state.redis.exists(key):
                    await state.redis.delete(key)
            
            print("✅ Test data cleaned up")
            
    except Exception as e:
        print(f"⚠️ Cleanup failed (this is usually okay): {e}")


async def main():
    """Run all tests."""
    print("🚀 Starting Redis parameter server tests...")
    print("=" * 60)
    
    tests = [
        test_basic_connection,
        test_robot_state,
        test_stroke_session,
        test_node_health,
        test_system_status,
        test_error_reporting,
        test_pub_sub,  # Run this last as it's most complex
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Cleanup
    await cleanup_test_data()
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Redis parameter server is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check Redis configuration and connectivity.")
        return 1


if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)