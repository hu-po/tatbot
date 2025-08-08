#!/usr/bin/env python3

import argparse
import signal
import sys
import threading
import time
from datetime import datetime

from node_monitor import NodeInfo, NodeMonitor
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn


class OpencodeTUI:
    def __init__(self, update_interval: int = 5):
        self.monitor = NodeMonitor()
        self.update_interval = update_interval
        self.console = Console()
        self.running = True
        self.last_update = None
        self.force_refresh = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.running = False
        self.monitor.close()
        sys.exit(0)
    
    def create_header(self) -> Text:
        header_text = Text()
        header_text.append("🤖 OPENCODE NODES ", style="bold magenta")
        header_text.append(f"| {datetime.now().strftime('%H:%M:%S')} ", style="dim")
        header_text.append(f"| Interval: {self.update_interval}s ", style="dim")
        header_text.append("| [q]uit [r]efresh [i]nterval- [I]nterval+", style="dim cyan")
        return Align.center(header_text)
    
    def create_progress_bar(self, value: float, max_value: float = 100, width: int = 20) -> str:
        percentage = min(100, max(0, value))
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage:3.0f}%"

    def create_node_panel(self, node: NodeInfo) -> Panel:
        lines = []
        
        if node.is_online:
            if node.cpu_load is not None:
                cpu_color = "green" if node.cpu_load <= 50 else "yellow" if node.cpu_load <= 80 else "red"
                cpu_src = "[R]" if node.cpu_count_source == "remote" else "[M]" if node.cpu_count_source == "meta" else ""
                cpu_bar = self.create_progress_bar(node.cpu_load, width=15)
                lines.append(Text(f"CPU: {node.cpu_count or '?'}c {cpu_bar} {cpu_src}", style=cpu_color))
            elif node.cpu_count:
                lines.append(Text(f"CPU: {node.cpu_count} cores [M]", style="dim"))
            
            if node.memory_usage:
                mem_p = node.memory_usage['percent']
                mem_color = "green" if mem_p <= 50 else "yellow" if mem_p <= 80 else "red"
                mem_bar = self.create_progress_bar(mem_p, width=15)
                lines.append(Text(f"MEM: {node.memory_usage['used_gb']:.0f}/{node.memory_usage['total_gb']:.0f}G", style=mem_color))
                lines.append(Text(f"     {mem_bar}", style=mem_color))
            
            if node.gpu_info:
                gpu_src = "[R]" if node.gpu_source == "remote" else "[M]" if node.gpu_source == "meta" else ""
                
                if node.gpu_info.get('gpus') and len(node.gpu_info['gpus']) > 0:
                    total_used = node.gpu_info.get('total_used_mb', 0)
                    total_mem = node.gpu_info.get('total_memory_mb', 0)
                    gpu_count = node.gpu_info.get('gpu_count', 1)
                    
                    if total_mem > 0:
                        gpu_percent = (total_used / total_mem) * 100
                        gpu_bar = self.create_progress_bar(gpu_percent, width=15)
                        gpu_str = f"GPU: {total_used/1024:.1f}/{total_mem/1024:.1f}G"
                        if gpu_count > 1:
                            gpu_str += f" x{gpu_count}"
                        lines.append(Text(f"{gpu_str} {gpu_src}", style="cyan"))
                        lines.append(Text(f"     {gpu_bar}", style="cyan"))
                    
                    gpu = node.gpu_info['gpus'][0]
                    temp_color = "green" if gpu['temperature_c'] <= 70 else "yellow" if gpu['temperature_c'] <= 85 else "red"
                    lines.append(Text(f"🌡️ {gpu['temperature_c']:.0f}°C | ⚡ {gpu['utilization_percent']:.0f}%", style=temp_color))
                    
                elif node.gpu_info.get('meta_only'):
                    total_mem = node.gpu_info.get('total_memory_mb', 0)
                    gpu_count = node.gpu_info.get('gpu_count', 0)
                    gpu_str = f"GPU: {total_mem/1024:.1f}G"
                    if gpu_count > 1:
                        gpu_str += f" x{gpu_count}"
                    lines.append(Text(f"{gpu_str} [M]", style="dim cyan"))
        else:
            lines.append(Text("OFFLINE", style="red bold"))
            if node.cpu_count:
                lines.append(Text(f"CPU: {node.cpu_count} cores [M]", style="dim"))
            if node.gpu_info and node.gpu_info.get('meta_only'):
                total_mem = node.gpu_info.get('total_memory_mb', 0)
                gpu_count = node.gpu_info.get('gpu_count', 0)
                gpu_str = f"GPU: {total_mem/1024:.1f}G"
                if gpu_count > 1:
                    gpu_str += f" x{gpu_count}"
                lines.append(Text(f"{gpu_str} [M]", style="dim cyan"))
        
        status = "🟢" if node.is_online else "🔴"
        return Panel(
            "\n".join(str(l) for l in lines),
            title=f"{status} {node.emoji} {node.name}",
            border_style="green" if node.is_online else "red",
            box=box.ROUNDED,
            height=9
        )
    
    def create_summary_table(self) -> Table:
        table = Table(box=box.ROUNDED)
        table.add_column("Node", style="cyan", width=12)
        table.add_column("Status", justify="center", width=8)
        table.add_column("CPU", justify="right", width=15)
        table.add_column("Memory", justify="right", width=15) 
        table.add_column("GPU", justify="right", width=20)
        table.add_column("Source", justify="center", width=8)
        
        for node in self.monitor.nodes:
            status = "✅ ONLINE" if node.is_online else "❌ OFFLINE"
            status_color = "green" if node.is_online else "red"
            
            if node.cpu_load:
                cpu = f"{node.cpu_load:.0f}%@{node.cpu_count}c" if node.cpu_count else f"{node.cpu_load:.0f}%"
            elif node.cpu_count:
                cpu = f"{node.cpu_count}c"
            else:
                cpu = "-"
            
            mem = f"{node.memory_usage['percent']:.0f}%" if node.memory_usage else "-"
            
            if node.gpu_info:
                if node.gpu_info.get('gpus'):
                    total_used = node.gpu_info.get('total_used_mb', 0)
                    total_mem = node.gpu_info.get('total_memory_mb', 0)
                    gpu = f"{total_used/1024:.1f}/{total_mem/1024:.1f}G"
                elif node.gpu_info.get('meta_only'):
                    total_mem = node.gpu_info.get('total_memory_mb', 0)
                    gpu = f"{total_mem/1024:.1f}G"
                else:
                    gpu = "✓"
            else:
                gpu = "-"
            
            sources = []
            if node.cpu_count_source == "remote":
                sources.append("R")
            elif node.cpu_count_source == "meta":
                sources.append("M")
            if node.gpu_source == "remote":
                if "R" not in sources:
                    sources.append("R")
            elif node.gpu_source == "meta":
                if "M" not in sources:
                    sources.append("M")
            source = "/".join(sources) if sources else "-"
            
            table.add_row(
                f"{node.emoji} {node.name}",
                Text(status, style=status_color),
                cpu,
                mem,
                gpu,
                source
            )
        
        return table
    
    def create_layout(self) -> Layout:
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=2),
            Layout(name="body"),
            Layout(name="footer", size=4)
        )
        
        layout["header"].update(self.create_header())
        
        body = Layout()
        rows = []
        nodes_per_row = 3
        
        for i in range(0, len(self.monitor.nodes), nodes_per_row):
            row = Layout()
            row_nodes = self.monitor.nodes[i:i+nodes_per_row]
            
            if len(row_nodes) == nodes_per_row:
                row.split_row(
                    *[Layout(self.create_node_panel(node)) for node in row_nodes]
                )
            elif len(row_nodes) == 2:
                row.split_row(
                    Layout(self.create_node_panel(row_nodes[0])),
                    Layout(self.create_node_panel(row_nodes[1])),
                    Layout()
                )
            else:
                row.update(self.create_node_panel(row_nodes[0]))
            
            rows.append(row)
        
        if rows:
            body.split(*rows)
        
        layout["body"].update(body)
        
        footer_panel = Panel(
            self.create_summary_table(),
            title="📊 Node Summary ([R]=Remote, [M]=Metadata)",
            border_style="dim",
            box=box.ROUNDED
        )
        layout["footer"].update(footer_panel)
        
        return layout
    
    def update_data(self):
        while self.running:
            self.monitor.update_all_nodes()
            self.last_update = time.time()
            
            # Check for force refresh or wait for interval
            elapsed = 0
            while elapsed < self.update_interval and self.running:
                if self.force_refresh:
                    self.force_refresh = False
                    break
                time.sleep(0.1)
                elapsed += 0.1
    
    def handle_input(self):
        import sys
        import select
        import tty
        import termios
        
        if not sys.stdin.isatty():
            return
            
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() == 'q':
                        self.running = False
                        break
                    elif ch.lower() == 'r':
                        self.force_refresh = True
                    elif ch.lower() == 'i':
                        self.update_interval = max(1, self.update_interval - 1)
                    elif ch.upper() == 'I':
                        self.update_interval = min(60, self.update_interval + 1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def run(self):
        update_thread = threading.Thread(target=self.update_data, daemon=True)
        update_thread.start()
        
        input_thread = threading.Thread(target=self.handle_input, daemon=True)
        input_thread.start()
        
        with Live(self.create_layout(), refresh_per_second=2, console=self.console) as live:
            while self.running:
                try:
                    live.update(self.create_layout())
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    self.running = False
                    break
        
        self.monitor.close()


def main():
    parser = argparse.ArgumentParser(description="Opencode Node Monitor TUI")
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Update interval in seconds (default: 5)"
    )
    
    args = parser.parse_args()
    
    tui = OpencodeTUI(update_interval=args.interval)
    tui.run()


if __name__ == "__main__":
    main()