#!/usr/bin/env python3

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from node_monitor import NodeInfo, NodeMonitor
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "tui.log"

# Configure logging for TUI
logger = logging.getLogger('TatbotTUI')
logger.setLevel(logging.DEBUG)

# Add file handler if not already present
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class TatbotTUI:
    def __init__(self, update_interval: int = 5):
        logger.info(f"Initializing TatbotTUI with {update_interval}s update interval")
        try:
            self.monitor = NodeMonitor()
            self.update_interval = update_interval
            self.console = Console()
            self.running = True
            
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("TatbotTUI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TUI: {e}", exc_info=True)
            raise
    
    def _signal_handler(self, signum, frame):
        self.running = False
        self.monitor.close()
        sys.exit(0)
    
    def create_header(self) -> Text:
        header_text = Text()
        header_text.append("🤖 TATBOT NODES ", style="bold cyan")
        header_text.append(f"| {datetime.now().strftime('%H:%M:%S')}", style="dim")
        return Align.center(header_text)
    
    def create_node_panel(self, node: NodeInfo) -> Panel:
        """Create a simplified node panel"""
        try:
            lines = []
            stats = node.stats
            
            if not stats:
                logger.warning(f"Node {node.name} has no stats")
                lines.append(Text("NO STATS", style="red"))
            elif stats.online:
                # CPU info
                if stats.cpu_load is not None:
                    cpu_color = "green" if stats.cpu_load <= 50 else "yellow" if stats.cpu_load <= 80 else "red"
                    source = "[R]" if stats.data_source == "remote" else "[M]"
                    lines.append(Text(f"CPU: {stats.cpu_load:.0f}% ({stats.cpu_cores}c) {source}", style=cpu_color))
                
                # Memory info
                if stats.memory_used_gb is not None and stats.memory_total_gb is not None:
                    mem_color = "green" if stats.memory_percent <= 50 else "yellow" if stats.memory_percent <= 80 else "red"
                    lines.append(Text(f"MEM: {stats.memory_used_gb:.1f}/{stats.memory_total_gb:.1f}G", style=mem_color))
                
                # GPU info
                if stats.gpu_used_gb is not None and stats.gpu_total_gb is not None:
                    gpu_color = "cyan"
                    gpu_line = f"GPU: {stats.gpu_used_gb:.1f}/{stats.gpu_total_gb:.1f}G"
                    if stats.gpu_count and stats.gpu_count > 1:
                        gpu_line += f" x{stats.gpu_count}"
                    lines.append(Text(gpu_line, style=gpu_color))
                    
                    # Temperature and utilization
                    if stats.gpu_temp and stats.gpu_util:
                        lines.append(Text(f"     {stats.gpu_temp:.0f}°C  {stats.gpu_util:.0f}%", style="dim"))
                elif stats.gpu_total_gb is not None:  # Metadata only
                    gpu_line = f"GPU: {stats.gpu_total_gb:.1f}G [M]"
                    if stats.gpu_count and stats.gpu_count > 1:
                        gpu_line += f" x{stats.gpu_count}"
                    lines.append(Text(gpu_line, style="dim cyan"))
            
            else:
                lines.append(Text("OFFLINE", style="red"))
                # Show metadata for offline nodes
                if stats.cpu_cores:
                    lines.append(Text(f"CPU: {stats.cpu_cores}c [M]", style="dim"))
                if stats.gpu_total_gb:
                    gpu_line = f"GPU: {stats.gpu_total_gb:.1f}G [M]"
                    if stats.gpu_count and stats.gpu_count > 1:
                        gpu_line += f" x{stats.gpu_count}"
                    lines.append(Text(gpu_line, style="dim cyan"))
            
            # Show errors if any
            if stats and stats.error:
                lines.append(Text(f"Error: {stats.error}", style="red dim"))
            
            status_emoji = "🟢" if stats and stats.online else "🔴"
            title = f"{status_emoji} {node.emoji}{node.name}"
            border_style = "green" if stats and stats.online else "red"
            
            return Panel(
                "\n".join(str(line) for line in lines),
                title=title,
                border_style=border_style,
                box=box.MINIMAL,
                height=6
            )
        except Exception as e:
            logger.error(f"Failed to create panel for {node.name}: {e}", exc_info=True)
            return Panel(
                f"Panel Error: {e}",
                title=f"🔴 {node.emoji}{node.name}",
                border_style="red",
                box=box.MINIMAL,
                height=6
            )
    
    def create_summary_table(self) -> Table:
        """Create simplified summary table"""
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Node", style="cyan", width=12)
        table.add_column("Status", justify="center", width=6)
        table.add_column("CPU", justify="right", width=15)
        table.add_column("Memory", justify="right", width=15)
        table.add_column("GPU", justify="right", width=15)
        
        for node in self.monitor.nodes:
            stats = node.stats
            
            # Status
            status = "✓" if stats.online else "✗"
            status_color = "green" if stats.online else "red"
            
            # CPU
            if stats.cpu_load is not None:
                cpu = f"{stats.cpu_load:.0f}% ({stats.cpu_cores}c)"
            elif stats.cpu_cores:
                cpu = f"({stats.cpu_cores}c)"
            else:
                cpu = "-"
            
            # Memory
            if stats.memory_used_gb is not None and stats.memory_total_gb is not None:
                memory = f"{stats.memory_used_gb:.1f}/{stats.memory_total_gb:.1f}G"
            else:
                memory = "-"
            
            # GPU
            if stats.gpu_used_gb is not None and stats.gpu_total_gb is not None:
                gpu = f"{stats.gpu_used_gb:.1f}/{stats.gpu_total_gb:.1f}G"
            elif stats.gpu_total_gb is not None:
                gpu = f"{stats.gpu_total_gb:.1f}G [M]"
            else:
                gpu = "-"
            
            table.add_row(
                f"{node.emoji}{node.name}",
                Text(status, style=status_color),
                cpu,
                memory,
                gpu
            )
        
        return table
    
    def create_layout(self) -> Layout:
        """Create simplified layout"""
        try:
            layout = Layout()
            
            layout.split(
                Layout(name="header", size=1),
                Layout(name="body"),
                Layout(name="footer", size=len(self.monitor.nodes) + 3)
            )
            
            layout["header"].update(self.create_header())
            
            # Create grid of node panels
            body = Layout()
            rows = []
            nodes_per_row = 3
            
            for i in range(0, len(self.monitor.nodes), nodes_per_row):
                row = Layout()
                row_nodes = self.monitor.nodes[i:i+nodes_per_row]
                
                if len(row_nodes) == nodes_per_row:
                    row.split_row(*[Layout(self.create_node_panel(node)) for node in row_nodes])
                else:
                    # Handle incomplete rows
                    panels = [Layout(self.create_node_panel(node)) for node in row_nodes]
                    panels.extend([Layout() for _ in range(nodes_per_row - len(row_nodes))])
                    row.split_row(*panels)
                
                rows.append(row)
            
            if rows:
                body.split(*rows)
            
            layout["body"].update(body)
            
            # Footer with summary
            footer_panel = Panel(
                self.create_summary_table(),
                title="Summary ([R]=Remote data, [M]=Metadata)",
                border_style="dim",
                box=box.SIMPLE
            )
            layout["footer"].update(footer_panel)
            
            return layout
        except Exception as e:
            logger.error(f"Failed to create layout: {e}", exc_info=True)
            # Return minimal error layout
            error_layout = Layout()
            error_layout.update(Panel(f"Layout Error: {e}", style="red"))
            return error_layout
    
    def update_data(self):
        """Background data update loop"""
        while self.running:
            self.monitor.update_all_nodes()
            time.sleep(self.update_interval)
    
    def run(self):
        """Main TUI loop"""
        try:
            logger.info("Starting TUI")
            
            # Start background update thread
            update_thread = threading.Thread(target=self.update_data, daemon=True)
            update_thread.start()
            
            # Initial update
            logger.info("Performing initial node update")
            self.monitor.update_all_nodes()
            
            logger.info("Starting Rich Live display")
            with Live(self.create_layout(), refresh_per_second=1, console=self.console) as live:
                while self.running:
                    try:
                        live.update(self.create_layout())
                        time.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("Received keyboard interrupt")
                        break
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}", exc_info=True)
                        time.sleep(1)  # Prevent tight error loop
            
            logger.info("Closing TUI")
            self.monitor.close()
            
        except Exception as e:
            logger.error(f"Fatal error in TUI run: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description="Tatbot Node Monitor TUI")
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Update interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug console output"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        # Add console handler for debug mode
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logging.getLogger('NodeMonitor').addHandler(console_handler)
    
    try:
        logger.info(f"Starting Tatbot TUI with {args.interval}s interval")
        tui = TatbotTUI(update_interval=args.interval)
        tui.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error occurred. Check log file at: {LOG_PATH}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nUnhandled exception: {e}")
        print(f"Check log file at: {LOG_PATH}")
        sys.exit(1)