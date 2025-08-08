from __future__ import annotations

import curses
import time
from typing import List, Optional

# Support both package execution and direct script execution from this folder
try:  # when executed as a package (if the folder name were importable)
    from .nodes_config import Node, parse_nodes_config  # type: ignore
    from .remote_stats import NodeStats, get_node_stats  # type: ignore
    from .nodes_meta import NodeMeta, parse_nodes_meta  # type: ignore
except Exception:
    # When run directly as: python src/tui-cursor/tui.py
    import importlib.util
    import os
    import sys

    _BASE_DIR = os.path.dirname(__file__)

    def _import_by_path(module_name: str, filename: str):
        file_path = os.path.join(_BASE_DIR, filename)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {module_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    nodes_config = _import_by_path("nodes_config", "nodes_config.py")
    remote_stats = _import_by_path("remote_stats", "remote_stats.py")
    nodes_meta = _import_by_path("nodes_meta", "nodes_meta.py")

    parse_nodes_config = nodes_config.parse_nodes_config
    Node = nodes_config.Node
    get_node_stats = remote_stats.get_node_stats
    NodeStats = remote_stats.NodeStats
    parse_nodes_meta = nodes_meta.parse_nodes_meta
    NodeMeta = nodes_meta.NodeMeta

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "conf", "nodes.yaml")
# Use metadata file that lives alongside this TUI implementation so users
# don't need to modify their existing config directory.
META_PATH = os.path.join(os.path.dirname(__file__), "nodes_meta.yaml")
REFRESH_SECS = 2.0


def format_cpu(stats: NodeStats, meta: 'NodeMeta | None') -> str:
    if not stats.online:
        return f"- ({meta.cpu_cores} cores) [M]" if meta and meta.cpu_cores else "-"
    if not stats.cpu:
        return f"n/a ({meta.cpu_cores} cores) [M]" if meta and meta.cpu_cores else "n/a"
    logical = getattr(stats.cpu, 'logical_cores', None)
    phys = getattr(stats.cpu, 'physical_cores', None)
    cores_disp = phys or logical or (meta.cpu_cores if meta else None)
    suffix = f" ({cores_disp} cores)" if cores_disp else ""
    pct = getattr(stats.cpu, 'percent_1m', None)
    pct_txt = f" {pct:.0f}%" if pct is not None else ""
    source = "[R]" if logical is not None else ("[M]" if (cores_disp and meta and meta.cpu_cores == cores_disp) else "")
    return f"{stats.cpu.load_1:.2f} {stats.cpu.load_5:.2f} {stats.cpu.load_15:.2f}{pct_txt}{suffix} {source}".strip()


def format_gpu(stats: NodeStats, meta: 'NodeMeta | None') -> str:
    if not stats.online:
        if meta and meta.gpu_total_mb:
            gcount = f"/{meta.gpu_count}g" if (meta.gpu_count and meta.gpu_count > 1) else ""
            total_gib = meta.gpu_total_mb / 1024.0
            return f"-/ {total_gib:.1f} GiB{gcount} [M]"
        return "-"
    if not stats.gpu:
        if meta and meta.gpu_total_mb:
            gcount = f"/{meta.gpu_count}g" if (meta.gpu_count and meta.gpu_count > 1) else ""
            total_gib = meta.gpu_total_mb / 1024.0
            return f"n/a/ {total_gib:.1f} GiB{gcount} [M]"
        return "n/a"
    gcount = f"/{stats.gpu.gpu_count}g" if (stats.gpu.gpu_count and stats.gpu.gpu_count > 1) else ""
    total = stats.gpu.mem_total_mb or (meta.gpu_total_mb if meta else None)
    if total is None:
        used_gib = stats.gpu.mem_used_mb / 1024.0
        return f"{used_gib:.1f}/ ? GiB"
    used_gib = stats.gpu.mem_used_mb / 1024.0
    total_gib = total / 1024.0
    source = "[R]" if stats.gpu.mem_total_mb else ("[M]" if (meta and meta.gpu_total_mb) else "")
    # If there are multiple GPUs, show a compact per-GPU suffix like [0:12/24,1:0/24] GiB
    per_gpu = ""
    try:
        devices = getattr(stats.gpu, 'devices', None)
        if devices and len(devices) > 1:
            parts = []
            for dev in devices:
                parts.append(f"{dev.index}:{dev.mem_used_mb/1024.0:.0f}/{dev.mem_total_mb/1024.0:.0f}")
            per_gpu = f" [{','.join(parts)}]"
    except Exception:
        per_gpu = ""
    return f"{used_gib:.1f}/ {total_gib:.1f} GiB{gcount}{per_gpu} {source}".strip()


def draw_header(stdscr, title: str) -> None:
    height, width = stdscr.getmaxyx()
    header = f" {title} (q: quit, r: refresh, i: interval) "
    stdscr.attron(curses.color_pair(2))
    try:
        stdscr.addstr(0, 0, header[: max(0, width - 1)])
        if len(header) < width:
            stdscr.addstr(0, len(header), " " * max(0, width - len(header) - 1))
    except curses.error:
        pass
    stdscr.attroff(curses.color_pair(2))


def draw_table(stdscr, nodes: List[Node], node_stats: dict[str, NodeStats], metas: dict[str, NodeMeta], last_updated: float, interval: float) -> None:
    height, width = stdscr.getmaxyx()

    # Column widths
    col_name = 14
    col_ip = 15
    col_status = 5
    col_cpu = 26
    col_mem = 13
    col_gpu = max(10, width - (1 + col_name + 1 + col_ip + 1 + col_status + 1 + col_cpu + 1 + col_mem + 3))

    header_y = 2
    stdscr.attron(curses.A_BOLD)
    header_line = f"{'Node':<{col_name}} {'IP':<{col_ip}} {'On?':<{col_status}} {'CPU (1/5/15 & %)':<{col_cpu}} {'Mem':<{col_mem}} {'GPU':<{col_gpu}}"
    try:
        stdscr.addstr(header_y, 1, header_line[: max(0, width - 2)])
    except curses.error:
        pass
    stdscr.attroff(curses.A_BOLD)

    row_y = header_y + 2
    for node in nodes:
        stats = node_stats.get(node.name)
        meta = metas.get(node.name)
        status_txt = "on" if (stats and stats.online) else "off"
        color = curses.color_pair(3) if status_txt == "on" else curses.color_pair(1)

        stdscr.attron(color)
        # Prepare safe display values
        s = stats
        mem_txt = "-"
        if s and getattr(s, 'memory', None):
            m = s.memory
            mem_txt = f"{m.used_bytes/1e9:.1f}/{m.total_bytes/1e9:.1f}G ({m.percent:.0f}%)"
        cpu_txt = format_cpu(s, meta) if s else (f"- ({meta.cpu_cores} cores) [M]" if (meta and meta.cpu_cores) else "-")
        gpu_txt = format_gpu(s, meta) if s else (f"-/ {meta.gpu_total_mb/1024.0:.1f} GiB{'/' + str(meta.gpu_count) + 'g' if (meta and meta.gpu_count and meta.gpu_count>1) else ''} [M]" if (meta and meta.gpu_total_mb) else "-")
        line = f"{node.emoji} {node.name:<{col_name-2}} {node.ip:<{col_ip}} {status_txt:<{col_status}} {cpu_txt:<{col_cpu}} {mem_txt:<{col_mem}} {gpu_txt:<{col_gpu}}"
        try:
            stdscr.addstr(row_y, 1, line[: max(0, width - 2)])
        except curses.error:
            pass
        stdscr.attroff(color)
        row_y += 1
        if row_y >= height - 2:
            break

    # Footer
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_updated))
    footer = f" Last updated: {ts} | Interval: {interval:.0f}s "
    stdscr.attron(curses.color_pair(2))
    try:
        stdscr.addstr(height - 1, 0, footer[: max(0, width - 1)])
        if len(footer) < width:
            stdscr.addstr(height - 1, len(footer), " " * max(0, width - len(footer) - 1))
    except curses.error:
        pass
    stdscr.attroff(curses.color_pair(2))


def main(stdscr, interval: float = 2.0):
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(3, curses.COLOR_GREEN, -1)

    nodes = parse_nodes_config(CONFIG_PATH)
    metas = parse_nodes_meta(META_PATH)
    node_stats: dict[str, NodeStats] = {n.name: get_node_stats(n.ip, n.user) for n in nodes}
    last_updated = time.time()

    while True:
        stdscr.erase()
        draw_header(stdscr, "Tatbot Nodes")
        draw_table(stdscr, nodes, node_stats, metas, last_updated, interval)
        stdscr.refresh()

        # Non-blocking key read with timeout for refresh
        stdscr.timeout(int(interval * 1000))
        ch = stdscr.getch()
        if ch in (ord('q'), ord('Q')):
            break
        if ch in (ord('r'), ord('R')):
            # immediate refresh
            pass
        if ch in (ord('i'), ord('I')):
            # Decrease interval with 'i' key down to a floor
            interval = max(1.0, interval - 1.0)

        # Refresh stats (basic parallelism using threads for snappier UI)
        import concurrent.futures

        def fetch(n: Node):
            return n.name, get_node_stats(n.ip, n.user)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(2, len(nodes)))) as pool:
            for name, stats in pool.map(fetch, nodes):
                node_stats[name] = stats
        last_updated = time.time()


if __name__ == "__main__":
    # Support optional interval from env var for quick tuning
    interval_env = os.environ.get("TATBOT_TUI_INTERVAL")
    try:
        interval = float(interval_env) if interval_env else 2.0
    except Exception:
        interval = 2.0
    curses.wrapper(lambda scr: main(scr, interval=interval))
