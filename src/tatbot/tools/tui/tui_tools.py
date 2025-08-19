"""TUI monitoring tools for MCP integration."""

import asyncio
import os
import signal
import subprocess
import sys
from typing import Dict

import psutil

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.tui.models import (
    ListTUIMonitorsInput,
    ListTUIMonitorsOutput,
    StartTUIMonitorInput,
    StartTUIMonitorOutput,
    StopTUIMonitorInput,
    StopTUIMonitorOutput,
)
from tatbot.utils.log import get_logger

log = get_logger("tools.tui", "📺")

# Global registry for background monitors
_monitor_processes: Dict[int, subprocess.Popen] = {}


@tool(
    name="start_tui_monitor",
    nodes=["rpi1", "rpi2", "oop", "ook"],
    description="Start TUI system monitor for real-time tatbot system visualization",
    input_model=StartTUIMonitorInput,
    output_model=StartTUIMonitorOutput,
)
async def start_tui_monitor_tool(input_data: StartTUIMonitorInput, ctx: ToolContext):
    """
    Start the TUI system monitor for real-time visualization of tatbot state.
    
    This tool launches a terminal-based dashboard that displays:
    - System status and Redis connectivity
    - Real-time stroke progress from active sessions
    - Node health and connectivity status
    - Recent system events and errors
    
    Parameters:
    - refresh_rate: Update frequency in seconds (0.5-10.0)
    - background: Run in background as detached process
    
    Returns:
    - success: Whether the monitor started successfully
    - message: Status message
    - process_id: Process ID if running in background
    """
    yield {"progress": 0.1, "message": "Starting TUI system monitor..."}
    
    try:
        if input_data.background:
            # Start as background process
            yield {"progress": 0.3, "message": "Launching monitor in background..."}
            
            # Create command to run monitor
            python_path = sys.executable
            module_path = "tatbot.tui.monitor"
            
            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = ":".join(sys.path)
            
            # Start process
            process = subprocess.Popen(
                [python_path, "-m", module_path, "--refresh-rate", str(input_data.refresh_rate)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent
            )
            
            # Register process
            _monitor_processes[process.pid] = process
            
            # Verify it started
            await asyncio.sleep(1)
            if process.poll() is None:  # Still running
                yield StartTUIMonitorOutput(
                    success=True,
                    message=f"✅ TUI monitor started in background (PID: {process.pid})",
                    process_id=process.pid
                )
            else:
                yield StartTUIMonitorOutput(
                    success=False,
                    message=f"❌ Monitor process exited immediately (return code: {process.returncode})"
                )
        else:
            # Start in foreground - import and run directly
            yield {"progress": 0.3, "message": "Starting monitor in foreground..."}
            
            from tatbot.tui.monitor import TatbotMonitor
            
            monitor = TatbotMonitor(refresh_rate=input_data.refresh_rate)
            
            yield {"progress": 0.5, "message": "Monitor initialized, starting display..."}
            
            # This will block until user exits with Ctrl+C
            try:
                await monitor.run()
                yield StartTUIMonitorOutput(
                    success=True,
                    message="✅ TUI monitor completed successfully"
                )
            except KeyboardInterrupt:
                yield StartTUIMonitorOutput(
                    success=True,
                    message="✅ TUI monitor stopped by user"
                )
            
    except ImportError as e:
        yield StartTUIMonitorOutput(
            success=False,
            message=f"❌ Missing TUI dependencies: {e}. Install with: uv pip install -e .[tui]"
        )
    except Exception as e:
        log.error(f"Error starting TUI monitor: {e}")
        yield StartTUIMonitorOutput(
            success=False,
            message=f"❌ Failed to start TUI monitor: {e}"
        )


@tool(
    name="stop_tui_monitor",
    nodes=["rpi1", "rpi2", "oop", "ook"],
    description="Stop running TUI system monitor",
    input_model=StopTUIMonitorInput,
    output_model=StopTUIMonitorOutput,
)
async def stop_tui_monitor_tool(input_data: StopTUIMonitorInput, ctx: ToolContext):
    """
    Stop running TUI system monitor processes.
    
    This tool can stop specific monitor processes by PID or stop all
    running TUI monitors on the system.
    
    Parameters:
    - process_id: Specific process ID to stop (optional)
    
    Returns:
    - success: Whether the operation succeeded
    - message: Status message
    - stopped_processes: Number of processes stopped
    """
    yield {"progress": 0.1, "message": "Looking for TUI monitor processes..."}
    
    try:
        stopped_count = 0
        
        if input_data.process_id:
            # Stop specific process
            yield {"progress": 0.3, "message": f"Stopping monitor process {input_data.process_id}..."}
            
            if input_data.process_id in _monitor_processes:
                process = _monitor_processes[input_data.process_id]
                try:
                    process.terminate()
                    process.wait(timeout=5)  # Wait up to 5 seconds
                    del _monitor_processes[input_data.process_id]
                    stopped_count = 1
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if timeout
                    del _monitor_processes[input_data.process_id]
                    stopped_count = 1
            else:
                # Try to kill process by PID directly
                try:
                    os.kill(input_data.process_id, signal.SIGTERM)
                    stopped_count = 1
                except ProcessLookupError:
                    pass  # Process doesn't exist
        else:
            # Stop all monitor processes
            yield {"progress": 0.3, "message": "Stopping all TUI monitor processes..."}
            
            # Stop registered processes
            for pid, process in list(_monitor_processes.items()):
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    stopped_count += 1
                except subprocess.TimeoutExpired:
                    process.kill()
                    stopped_count += 1
                except:
                    pass
                finally:
                    del _monitor_processes[pid]
            
            # Find and stop any other monitor processes
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'tatbot.tui.monitor' in ' '.join(cmdline):
                        try:
                            proc.terminate()
                            proc.wait(timeout=5)
                            stopped_count += 1
                        except:
                            try:
                                proc.kill()
                                stopped_count += 1
                            except:
                                pass
            except:
                pass  # psutil might not be available
        
        if stopped_count > 0:
            yield StopTUIMonitorOutput(
                success=True,
                message=f"✅ Stopped {stopped_count} monitor process(es)",
                stopped_processes=stopped_count
            )
        else:
            yield StopTUIMonitorOutput(
                success=True,
                message="ℹ️ No running monitor processes found",
                stopped_processes=0
            )
            
    except Exception as e:
        log.error(f"Error stopping TUI monitor: {e}")
        yield StopTUIMonitorOutput(
            success=False,
            message=f"❌ Failed to stop monitor: {e}",
            stopped_processes=0
        )


@tool(
    name="list_tui_monitors",
    nodes=["rpi1", "rpi2", "oop", "ook"],
    description="List running TUI system monitor processes",
    input_model=ListTUIMonitorsInput,
    output_model=ListTUIMonitorsOutput,
)
async def list_tui_monitors_tool(input_data: ListTUIMonitorsInput, ctx: ToolContext):
    """
    List all running TUI system monitor processes.
    
    This tool scans for active TUI monitor processes and provides
    information about their status and resource usage.
    
    Returns:
    - success: Whether the operation succeeded
    - message: Status message
    - monitors: List of running monitor processes with details
    """
    yield {"progress": 0.1, "message": "Scanning for TUI monitor processes..."}
    
    try:
        monitors = []
        
        # Check registered processes first
        for pid, process in list(_monitor_processes.items()):
            try:
                if process.poll() is None:  # Still running
                    monitors.append({
                        "pid": pid,
                        "status": "running",
                        "type": "registered",
                        "started": "unknown"
                    })
                else:
                    # Process died, clean up
                    del _monitor_processes[pid]
            except:
                # Clean up dead reference
                if pid in _monitor_processes:
                    del _monitor_processes[pid]
        
        # Scan for other monitor processes using psutil if available
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status', 'create_time', 'memory_info']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'tatbot.tui.monitor' in ' '.join(cmdline):
                        # Avoid duplicates
                        if not any(m['pid'] == proc.info['pid'] for m in monitors):
                            create_time = proc.info.get('create_time', 0)
                            if create_time:
                                import datetime
                                started = datetime.datetime.fromtimestamp(create_time).strftime("%H:%M:%S")
                            else:
                                started = "unknown"
                                
                            memory_mb = 0
                            if proc.info.get('memory_info'):
                                memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                                
                            monitors.append({
                                "pid": proc.info['pid'],
                                "status": proc.info.get('status', 'unknown'),
                                "type": "discovered",
                                "started": started,
                                "memory_mb": round(memory_mb, 1)
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            # psutil not available, use simple ps command
            try:
                import subprocess
                result = subprocess.run(
                    ["ps", "aux"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                for line in result.stdout.split('\n'):
                    if 'tatbot.tui.monitor' in line:
                        parts = line.split()
                        if len(parts) >= 11:
                            pid = int(parts[1])
                            if not any(m['pid'] == pid for m in monitors):
                                monitors.append({
                                    "pid": pid,
                                    "status": "running",
                                    "type": "ps_discovered",
                                    "started": parts[8] if len(parts) > 8 else "unknown"
                                })
            except:
                pass  # Fallback failed
        
        yield ListTUIMonitorsOutput(
            success=True,
            message=f"✅ Found {len(monitors)} running monitor process(es)",
            monitors=monitors
        )
        
    except Exception as e:
        log.error(f"Error listing TUI monitors: {e}")
        yield ListTUIMonitorsOutput(
            success=False,
            message=f"❌ Failed to list monitors: {e}",
            monitors=[]
        )