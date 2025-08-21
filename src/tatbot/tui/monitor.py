"""System monitoring TUI application for tatbot."""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tatbot.state.manager import StateManager
from tatbot.utils.log import get_logger
from tatbot.utils.node_config import load_node_ips, load_tui_config

log = get_logger("tui.monitor", "📺")


class TatbotMonitor:
    """Real-time TUI monitor for tatbot system state."""
    
    def __init__(self, redis_host: str = "eek", active_health_check: bool = True):
        self.console = Console()
        self.active_health_check = active_health_check
        
        # Load configuration
        self.tui_config = load_tui_config()
        self.refresh_rate = self.tui_config['refresh_rate']
        self.node_ips = load_node_ips()
        
        # StateManager reads Redis target from config; do not use environment.
        self.state_manager = StateManager(node_id="rpi1")
        self.running = False
        
        # Data storage
        self.system_status: Dict[str, Any] = {}
        self.current_stroke_session: Dict[str, Any] = {}
        self.recent_events: List[Dict[str, Any]] = []
        self.node_health: Dict[str, Any] = {}
        
        # UI components
        self.layout = Layout()
        self.setup_layout()
        
    def setup_layout(self) -> None:
        """Setup the TUI layout structure."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        self.layout["left"].split_column(
            Layout(name="system", ratio=1),
            Layout(name="strokes", ratio=1)
        )
        
        self.layout["right"].split_column(
            Layout(name="nodes", ratio=1),
            Layout(name="events", ratio=1)
        )
    
    def create_header(self) -> Panel:
        """Create header panel with system info."""
        title = Text("🤖 TATBOT SYSTEM MONITOR", style="bold magenta", justify="center")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        redis_status = "🟢 CONNECTED" if self.system_status.get("redis_connected", False) else "🔴 DISCONNECTED"
        nodes_online = self.system_status.get("nodes_online", 0)
        total_nodes = self.system_status.get("total_nodes", 0)
        
        status_line = Text()
        status_line.append(f"Redis: {redis_status}  ", style="bold")
        status_line.append(f"Nodes: {nodes_online}/{total_nodes}  ", style="bold")
        status_line.append(f"Updated: {timestamp}", style="dim")
        
        content = Text.assemble(title, "\n", status_line, justify="center")
        return Panel(content, box=box.DOUBLE, style="blue")
    
    def create_system_panel(self) -> Panel:
        """Create system status panel."""
        table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", justify="center")
        
        # Redis connection
        redis_connected = self.system_status.get("redis_connected", False)
        redis_icon = "🟢" if redis_connected else "🔴"
        # Display the actual target from StateManager (set at init)
        try:
            redis_host = getattr(self.state_manager.redis, 'host', 'eek')
            redis_port = getattr(self.state_manager.redis, 'port', 6379)
            redis_target = f"{redis_host}:{redis_port}"
        except Exception:
            redis_target = "eek:6379"
        table.add_row("Redis Server", redis_target, redis_icon)
        
        # Active sessions
        active_sessions = self.system_status.get("active_stroke_sessions", 0)
        session_icon = "🟡" if active_sessions > 0 else "⚪"
        table.add_row("Stroke Sessions", str(active_sessions), session_icon)
        
        # Nodes summary
        nodes_online = self.system_status.get("nodes_online", 0)
        total_nodes = self.system_status.get("total_nodes", 0)
        node_icon = "🟢" if nodes_online == total_nodes else "🟡" if nodes_online > 0 else "🔴"
        table.add_row("Nodes Online", f"{nodes_online}/{total_nodes}", node_icon)
        
        return Panel(table, title="📊 System Status", box=box.ROUNDED)
    
    def create_stroke_panel(self) -> Panel:
        """Create stroke progress panel."""
        if not self.current_stroke_session:
            content = Text("No active stroke session", style="dim", justify="center")
            return Panel(content, title="🎨 Stroke Progress", box=box.ROUNDED)
        
        progress_data = self.current_stroke_session.get("progress", {})
        
        # Session header
        scene_name = progress_data.get("scene_name", "unknown")
        node_id = progress_data.get("node_id", "unknown")
        
        session_text = Text()
        session_text.append("Session: ", style="bold")
        session_text.append(f"{scene_name}@{node_id}", style="cyan")
        
        # Progress info
        stroke_idx = progress_data.get("stroke_idx", 0)
        total_strokes = progress_data.get("total_strokes", 0)
        pose_idx = progress_data.get("pose_idx", 0)
        stroke_length = progress_data.get("stroke_length", 0)
        is_executing = progress_data.get("is_executing", False)
        
        content_parts = [session_text]
        
        # Progress bar representation
        if total_strokes > 0:
            progress_pct = (stroke_idx / total_strokes) * 100
            bar_length = 20
            filled = int((progress_pct / 100) * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            progress_text = Text()
            progress_text.append(f"  {bar} ", style="green" if is_executing else "dim")
            progress_text.append(f"{stroke_idx}/{total_strokes} ", style="bold")
            progress_text.append(f"({progress_pct:.1f}%)", style="dim")
            content_parts.append(progress_text)
            
            # Current pose info
            if stroke_length > 0 and pose_idx > 0:
                pose_pct = (pose_idx / stroke_length) * 100
                pose_text = Text()
                pose_text.append(f"  Pose: {pose_idx}/{stroke_length} ", style="white")
                pose_text.append(f"({pose_pct:.1f}%)", style="dim")
                content_parts.append(pose_text)
        
        # Status
        status_text = Text()
        status_text.append("  Status: ", style="bold")
        if is_executing:
            status_text.append("EXECUTING", style="bold green")
        else:
            status_text.append("IDLE", style="dim")
        content_parts.append(status_text)
            
        content = Text("\n").join(content_parts)
        return Panel(content, title="🎨 Stroke Progress", box=box.ROUNDED)
    
    def create_nodes_panel(self) -> Panel:
        """Create node health panel."""
        table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        table.add_column("Node", style="cyan", width=12)
        table.add_column("Status", justify="center", width=12)
        
        nodes_health = self.system_status.get("nodes_health", {})
        
        # Use configured nodes in order
        for node_id in sorted(self.node_ips.keys()):
            health_data = nodes_health.get(node_id)
            
            if health_data and health_data.get("is_reachable", False):
                status = "🟢 UP"
            elif health_data:
                status = "🔴 DOWN"
            else:
                status = "⚪ UNKNOWN"
            
            table.add_row(node_id, status)
        
        return Panel(table, title="🖥️  Node Health", box=box.ROUNDED)
    
    def create_events_panel(self) -> Panel:
        """Create recent events panel."""
        if not self.recent_events:
            content = Text("No recent events", style="dim", justify="center")
            return Panel(content, title="📡 Recent Events", box=box.ROUNDED)
        
        content_parts = []
        
        for event in self.recent_events[-10:]:  # Show last 10 events
            timestamp = event.get("timestamp", "")
            event_type = event.get("type", "unknown")
            node_id = event.get("node_id", "unknown")
            
            # Format timestamp
            time_str = "Unknown"
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
            
            # Event line
            event_text = Text()
            event_text.append(f"[{time_str}] ", style="dim")
            event_text.append(f"{node_id}: ", style="cyan")
            
            # Color code event types
            if event_type == "error":
                event_text.append(event_type.upper(), style="bold red")
            elif event_type in ["session_start", "progress_update"]:
                event_text.append(event_type.replace("_", " ").title(), style="green")
            elif event_type == "session_end":
                event_text.append(event_type.replace("_", " ").title(), style="yellow")
            else:
                event_text.append(event_type.replace("_", " ").title(), style="white")
            
            # Additional info
            if event_type == "progress_update":
                stroke_idx = event.get("stroke_idx", 0)
                total_strokes = event.get("total_strokes", 0)
                event_text.append(f" ({stroke_idx}/{total_strokes})", style="dim")
            elif event_type in ["session_start", "session_end"]:
                scene_name = event.get("scene_name", "")
                if scene_name:
                    event_text.append(f" ({scene_name})", style="dim")
            
            content_parts.append(event_text)
        
        content = Text("\n").join(content_parts)
        return Panel(content, title="📡 Recent Events", box=box.ROUNDED)
    
    def create_footer(self) -> Panel:
        """Create footer panel with controls."""
        controls = Text()
        controls.append("Controls: ", style="bold")
        controls.append("Ctrl+C", style="bold red")
        controls.append(" - Exit", style="white")
        
        return Panel(controls, box=box.SIMPLE, style="blue")
    
    async def check_node_health(self) -> Dict[str, Any]:
        """Actively check node health by pinging MCP servers."""
        import socket

        import aiohttp
        
        nodes_health = {}
        node_ips = self.node_ips
        
        current_time = datetime.now().isoformat()
        
        for node_id, ip in node_ips.items():
            node_health = {
                "node_id": node_id,
                "is_reachable": False,
                "timestamp": current_time,
                "check_method": "ping"
            }
            
            try:
                # Try to connect to MCP server first
                mcp_timeout = self.tui_config['health_check']['mcp_timeout']
                timeout = aiohttp.ClientTimeout(total=mcp_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        async with session.get(f"http://{ip}:8000/mcp") as response:
                            if response.status in [200, 405, 406]:  # MCP server responding
                                node_health["is_reachable"] = True
                                node_health["check_method"] = "mcp"
                    except:
                        # Fall back to basic ping
                        try:
                            # Simple socket connection test
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            ssh_timeout = self.tui_config['health_check']['ssh_timeout']
                            sock.settimeout(ssh_timeout)
                            result = sock.connect_ex((ip, 22))  # SSH port
                            if result == 0:
                                node_health["is_reachable"] = True
                                node_health["check_method"] = "ssh"
                            sock.close()
                        except:
                            pass
                            
            except Exception:
                pass
                
            nodes_health[node_id] = node_health
        
        return nodes_health

    async def update_data(self) -> None:
        """Update all monitoring data from Redis."""
        try:
            # Get system status from Redis
            self.system_status = await self.state_manager.get_system_status()
            
            if self.active_health_check:
                # Get active node health by actively checking nodes
                active_health = await self.check_node_health()
                
                # Merge Redis health data with active checks
                redis_health = self.system_status.get("nodes_health", {})
                for node_id, health in active_health.items():
                    if node_id in redis_health and redis_health[node_id]:
                        # Use Redis data if available (more detailed)
                        continue
                    else:
                        # Use our active check
                        redis_health[node_id] = health
                
                self.system_status["nodes_health"] = redis_health
                
                # Update node counts based on active checks
                online_count = sum(1 for h in active_health.values() if h["is_reachable"])
                self.system_status["nodes_online"] = online_count
                self.system_status["total_nodes"] = len(active_health)
            
            # Get most recent stroke session
            stroke_keys = await self.state_manager.redis.keys("stroke:progress:*")
            self.current_stroke_session = {}
            
            if stroke_keys:
                # Get the most recent session (sessions include timestamps)
                latest_key = max(stroke_keys)
                progress_data = await self.state_manager.redis.hget(latest_key, "progress")
                if progress_data:
                    try:
                        self.current_stroke_session = {
                            "progress": json.loads(progress_data)
                        }
                    except json.JSONDecodeError:
                        pass
            
            # Initialize with system start event if empty
            current_time = datetime.now().isoformat()
            if len(self.recent_events) == 0:
                self.recent_events.append({
                    "type": "system_start",
                    "node_id": "rpi1",
                    "timestamp": current_time
                })
                
        except Exception as e:
            log.error(f"Error updating data: {e}")
    
    async def listen_for_events(self) -> None:
        """Listen for real-time events from Redis pub/sub."""
        try:
            # Subscribe to all event channels
            channels = [
                "stroke:events:*",
                "robot:events",
                "error:events:*",
                "system:events"
            ]
            
            # Use pattern subscription for wildcard channels
            pubsub = self.state_manager.redis._redis.pubsub()
            await pubsub.psubscribe("stroke:events:*", "error:events:*")
            await pubsub.subscribe("robot:events", "system:events")
            
            log.info("📡 Listening for events...")
            
            async for message in pubsub.listen():
                if message["type"] in ["message", "pmessage"]:
                    try:
                        # Parse event data
                        if isinstance(message["data"], str):
                            event_data = json.loads(message["data"])
                        else:
                            event_data = message["data"]
                        
                        # Add to recent events
                        self.recent_events.append(event_data)
                        
                        # Keep only configured number of events
                        max_events = self.tui_config['display']['max_recent_events']
                        if len(self.recent_events) > max_events:
                            self.recent_events = self.recent_events[-max_events:]
                            
                    except (json.JSONDecodeError, TypeError) as e:
                        log.warning(f"Failed to parse event: {e}")
                        
        except Exception as e:
            log.error(f"Event listener error: {e}")
        finally:
            try:
                await pubsub.aclose()
            except:
                pass
    
    def render_layout(self) -> Layout:
        """Render the complete layout."""
        self.layout["header"].update(self.create_header())
        self.layout["system"].update(self.create_system_panel())
        self.layout["strokes"].update(self.create_stroke_panel())
        self.layout["nodes"].update(self.create_nodes_panel())
        self.layout["events"].update(self.create_events_panel())
        self.layout["footer"].update(self.create_footer())
        
        return self.layout
    
    async def run(self) -> None:
        """Run the monitoring TUI."""
        self.running = True
        event_task = None
        
        try:
            async with self.state_manager:
                log.info("🚀 Starting Tatbot System Monitor")
                
                # Start event listener in background
                event_task = asyncio.create_task(self.listen_for_events())
                
                with Live(self.render_layout(), console=self.console, refresh_per_second=1/self.refresh_rate) as live:
                    while self.running:
                        await self.update_data()
                        live.update(self.render_layout())
                        await asyncio.sleep(self.refresh_rate)
                        
        except KeyboardInterrupt:
            log.info("Monitor stopped by user")
        except Exception as e:
            # Check if it's a connection error
            if "Name or service not known" in str(e) or "Connection refused" in str(e):
                self.console.print(f"❌ Cannot connect to Redis server at {self.state_manager.redis.host}:{self.state_manager.redis.port}")
                self.console.print("💡 Ensure Redis is running on eek node or try:")
                self.console.print("   tatbot-monitor --redis-host 192.168.1.97")
                self.console.print("   ssh eek 'sudo redis-server /etc/redis/tatbot-redis.conf --daemonize yes'")
            else:
                log.error(f"Monitor error: {e}")
                raise
        finally:
            self.running = False
            # Cancel event listener
            if event_task:
                event_task.cancel()
                try:
                    await event_task
                except asyncio.CancelledError:
                    pass
    
    def stop(self) -> None:
        """Stop the monitor."""
        self.running = False


async def main() -> None:
    """Main entry point for the TUI monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tatbot System Monitor TUI")
    parser.add_argument(
        "--node-id",
        type=str,
        default="rpi1",
        help="Node ID for this monitor (default: rpi1)"
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="eek",
        help="Redis server host (default: eek, falls back to 192.168.1.97)"
    )
    parser.add_argument(
        "--no-active-health-check",
        action="store_true",
        help="Disable active node health checking (only use Redis data)"
    )
    
    args = parser.parse_args()
    
    monitor = TatbotMonitor(
        redis_host=args.redis_host,
        active_health_check=not args.no_active_health_check
    )
    monitor.state_manager.node_id = args.node_id
    
    await monitor.run()


def main_sync() -> None:
    """Synchronous entry point for script execution."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
