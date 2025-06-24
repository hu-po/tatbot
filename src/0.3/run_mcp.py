"""
TODO:

# server mvp
# mcp inspector
# cursor as client (settings add mcp server)
# add to cursor mcp directory https://cursor.directory/mcp

ideas:
- ping nodes
- git pull local tatbot repo, reinstall uv env

- send files to nodes
- run commands on nodes
- get plan data
- get images
- list available nodes
- get tatbot info
- get tech docs
- get configs
- generate plan
- sim control
- run plan on robot
- run camera calibration
- configure robot
- open up viz browser, chrome, turn on screen
- kill all python processes, kill all docker containers
- gradio frontend for tatbot MCP
- git pull on all machines
- uv env install on all machines
- distribute files to all machines
- ojo: start/stop containers, get CPU/GPU usage, pull latest pattern
- rpi1: pause/play live viz, set path and pose w/ live viz, reset live viz, open chrome
- trossen: reset/check realsenses, configure robot, run bot with CLI kwargs(0.3) 

"""
import concurrent.futures
from dataclasses import dataclass
import json
import logging
import os
import re
import tarfile
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from _log import get_logger, setup_log_with_config, print_config
from _net import NetworkManager

log = get_logger('run_mcp')

@dataclass
class MCPConfig:
    debug: bool = False
    """Enable debug logging."""
    transport: str = "streamable-http"
    """Transport type for MCP server."""

mcp = FastMCP("tatbot")
net = NetworkManager()

@mcp.resource("nodes://all")
def get_nodes() -> str:
    return "\n".join(f"{node.emoji} {node.name}" for node in net.nodes)

@mcp.tool(description="Tests connectivity to configured nodes and returns a status summary. If `nodes` is provided, only pings the specified nodes. Otherwise, pings all nodes.")
def ping_nodes(nodes: Optional[List[str]] = None) -> str:
    log.info(f"🔌 Pinging nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return error
    if not target_nodes:
        return "No nodes to ping."

    messages = []
    all_success = True

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_node = {
            executor.submit(net._test_node_connection, node): node for node in target_nodes
        }
        for future in concurrent.futures.as_completed(future_to_node):
            _name, success, message = future.result()
            messages.append(message)
            if not success:
                all_success = False

    header = "✅ All specified nodes are responding" if all_success else "❌ Some specified nodes are not responding"
    if not nodes:
        header = "✅ All nodes are responding" if all_success else "❌ Some nodes are not responding"

    return f"{header}:\n" + "\n".join(f"- {msg}" for msg in sorted(messages))

@mcp.tool(description="Runs 'git pull' on the tatbot repository on all configured nodes, then reinstalls the Python dependencies using uv. If `nodes` is provided, only updates the specified nodes.")
def update_nodes(nodes: Optional[List[str]] = None, timeout: float = 300.0) -> str:
    log.info(f"🔌 Updating nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return error
    if not target_nodes:
        return "No nodes to update."

    results = []

    for node in target_nodes:
        emoji = node.emoji

        log.info(f"{emoji} Updating {node.name} ({node.ip})")

        if net.is_local_node(node):
            results.append(f"{emoji} {node.name}: Skipped (local node)")
            continue

        try:
            client = net.get_ssh_client(node.ip, node.user)
            command = (
                "export PATH=\"$HOME/.local/bin:$PATH\" && "
                "git -C ~/tatbot pull && "
                "cd ~/tatbot/src/0.3 && "
                "deactivate >/dev/null 2>&1 || true && "
                "rm -rf .venv && "
                "rm -f uv.lock && "
                "uv venv && "
                f"uv pip install '{node.deps}'"
            )
            exit_code, out, err = net._run_remote_command(client, command, timeout=timeout)
            client.close()
            if exit_code == 0:
                results.append(f"{emoji} {node.name}: Success\n{out}")
            else:
                results.append(f"{emoji} {node.name}: Failed\n{err}")

        except Exception as e:
            results.append(f"{emoji} {node.name}: Exception occurred: {str(e)}")
            log.error(f"Failed to pull on {node.name}: {e}")

    return "\n\n".join(results)

@mcp.tool(description="For every node in config/nodes.yaml, report basic CPU/RAM usage. Falls back to 'unreachable' if SSH fails. If `nodes` is provided, only reports usage for the specified nodes.")
def node_cpu_ram_usage(nodes: Optional[List[str]] = None) -> dict[str, dict]:
    import psutil

    log.info(f"Getting usage for nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return {"error": error}

    report: dict[str, dict] = {}
    if not target_nodes:
        return report

    remote_nodes = [n for n in target_nodes if not net.is_local_node(n)]
    remote_node_names = [n.name for n in remote_nodes]

    # Handle local node
    for n in target_nodes:
        if net.is_local_node(n):
            report[n.name] = {
                "cpu_percent": psutil.cpu_percent(),
                "mem_percent": psutil.virtual_memory().percent,
            }

    # Handle remote nodes
    if remote_node_names:
        command = (
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "cd ~/tatbot/src/0.3 && "
            "uv run python - << 'EOF'\n"
            "import psutil, json, sys;"
            "print(json.dumps({'cpu_percent': psutil.cpu_percent(),"
            "'mem_percent': psutil.virtual_memory().percent}))\nEOF"
        )
        results = net.run_command_on_nodes(command, node_names=remote_node_names)

        for name, (exit_code, out, err) in results.items():
            if exit_code == 0:
                try:
                    report[name] = json.loads(out)
                except json.JSONDecodeError:
                    log.error(f"Failed to parse usage JSON from {name}: {out}")
                    report[name] = {"error": "invalid output"}
            elif exit_code == -1 and "Failed to connect" in err:
                report[name] = {"error": "unreachable"}
            else:
                log.error(f"Failed to get usage for {name}: {err}")
                report[name] = {"error": f"command failed: {err}"}

    return report

@mcp.tool(description="Powers off the specified nodes. Requires passwordless sudo for the 'poweroff' command.")
def poweroff_nodes(nodes: Optional[List[str]] = None) -> str:
    log.info(f"🔌 Powering off nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return error
    if not target_nodes:
        return "No nodes to power off."

    remote_nodes = [n for n in target_nodes if not net.is_local_node(n)]
    local_nodes = [n for n in target_nodes if net.is_local_node(n)]
    
    report = [f"⚠️ {node.emoji} {node.name}: Skipped (local node)." for node in local_nodes]

    if not remote_nodes:
        if report:
             return "\n".join(sorted(report))
        return "No remote nodes specified to power off."
    
    remote_node_names = [n.name for n in remote_nodes]
    remote_node_map = {n.name: n for n in remote_nodes}

    command = "nohup sudo poweroff > /dev/null 2>&1 &"
    # Use a short timeout because the command may not return on success
    results = net.run_command_on_nodes(command, node_names=remote_node_names, timeout=5.0)

    for name, (exit_code, out, err) in results.items():
        node = remote_node_map[name]
        # -1 exit code from our wrapper means an exception happened (like a timeout or connection drop)
        # which is expected if poweroff succeeds.
        if exit_code == -1 and ('session timed out' in err.lower() or 'socket is closed' in err.lower()):
            report.append(f"✅ {node.emoji} {name}: Power off command sent, connection lost as expected.")
        elif exit_code == -1 and ("Failed to connect" in err or 'timed out' in err.lower() or 'unable to connect' in err.lower()):
            report.append(f"🔌 {node.emoji} {name}: Already offline or unreachable.")
        elif exit_code == 0: # This might happen if poweroff returns immediately
             report.append(f"✅ {node.emoji} {name}: Power off command sent.")
        else:
            error_message = err or out
            report.append(f"❌ {node.emoji} {name}: Failed to power off. Exit code: {exit_code}, Error: {error_message}")
            log.error(f"Failed to power off {name}: Code={exit_code}, out={out}, err={err}")
    
    return "\n".join(sorted(report))

@mcp.tool(description="Turns on the visualization on the rpi1 node: pulls latest code, updates environment, kills existing Chromium and python3, launches viz process, waits for server, then launches Chromium. Logs all steps.")
def turn_on_viz() -> str:
    log.info(f"🔌 Turning on viz for rpi1: git pull, update uv env, kill old Chromium and python3, launch viz, wait for server, launch Chromium, with full logging.")

    rpi1_node = next((n for n in net.nodes if n.name == "rpi1"), None)
    if not rpi1_node:
        return "❌ rpi1 node not found in configuration."

    if net.is_local_node(rpi1_node):
        return "❌ rpi1 is the local node; this function is intended for remote execution."

    try:
        client = net.get_ssh_client(rpi1_node.ip, rpi1_node.user)
        script_content = (
            "#!/bin/bash\n"
            "echo whoami: > ~/chromium-viz.log\n"
            "whoami >> ~/chromium-viz.log\n"
            "echo env: >> ~/chromium-viz.log\n"
            "env >> ~/chromium-viz.log\n"
            "echo killing existing chromium... >> ~/chromium-viz.log\n"
            "pkill -f chromium-browser >> ~/chromium-viz.log 2>&1\n"
            "echo killing existing python3... >> ~/chromium-viz.log\n"
            "pkill -f python3 >> ~/chromium-viz.log 2>&1\n"
            "echo git pulling... >> ~/chromium-viz.log\n"
            "git -C ~/tatbot pull >> ~/chromium-viz.log 2>&1\n"
            "echo updating uv environment... >> ~/chromium-viz.log\n"
            "cd ~/tatbot/src/0.3\n"
            "deactivate >/dev/null 2>&1 || true\n"
            "rm -rf .venv\n"
            "rm -f uv.lock\n"
            "uv venv >> ~/chromium-viz.log 2>&1\n"
            "uv pip install '.[tag]' >> ~/chromium-viz.log 2>&1\n"
            "echo exporting display... >> ~/chromium-viz.log\n"
            "export DISPLAY=:0\n"
            "export XAUTHORITY=/home/rpi1/.Xauthority\n"
            "echo launching viz process... >> ~/chromium-viz.log\n"
            "source .venv/bin/activate\n"
            "setsid uv run _viz.py >> ~/chromium-viz.log 2>&1 &\n"
            "echo waiting for viser server on port 8080... >> ~/chromium-viz.log\n"
            "for i in {1..20}; do\n"
            "    if nc -z localhost 8080; then\n"
            "        echo viser server is up! >> ~/chromium-viz.log\n"
            "        break\n"
            "    fi\n"
            "    sleep 1\n"
            "done\n"
            "echo launching chromium... >> ~/chromium-viz.log\n"
            "setsid chromium-browser --kiosk http://localhost:8080 --disable-gpu >> ~/chromium-viz.log 2>&1 &\n"
        )
        # Write the script to a file on the remote machine
        sftp = client.open_sftp()
        with sftp.file('/home/rpi1/mcp_chromium_test.sh', 'w') as f:
            f.write(script_content)
        sftp.chmod('/home/rpi1/mcp_chromium_test.sh', 0o755)
        sftp.close()
        # Run the script with an interactive shell
        command = "bash -i ~/mcp_chromium_test.sh"
        exit_code, out, err = net._run_remote_command(client, command, timeout=80)
        client.close()
        if exit_code == 0:
            return f"✅ rpi1: Viz script executed (waits for server, launches Chromium after, kills python3).\n{out}"
        else:
            return f"❌ rpi1: Viz script failed.\n{err}"
    except Exception as e:
        log.error(f"Failed to run viz script on rpi1: {e}")
        return f"❌ rpi1: Exception occurred: {str(e)}"

@mcp.tool(description="Runs a scan on trossen-ai using bot_scan.py, then copies the resulting output directory to the local node, rpi1, and ook.")
def run_robot_scan() -> str:
    """
    1. Updates the tatbot repo and uv venv on trossen-ai.
    2. Runs `uv run bot_scan.py` on trossen-ai.
    3. Finds the new scan output directory (scan-<timestamp>).
    4. Copies the directory from trossen-ai to the local node.
    5. Distributes the directory to rpi1 and ook using transfer_files_to_nodes.
    """
    # Step 1: Prepare trossen-ai node
    trossen_node = next((n for n in net.nodes if n.name == "trossen-ai"), None)
    if not trossen_node:
        return "❌ trossen-ai node not found in configuration."
    if net.is_local_node(trossen_node):
        return "❌ trossen-ai is the local node; this function is intended for remote execution."

    remote_output_dir = "~/tatbot/output/record"
    scan_dir_pattern = r"scan-\d{4}y-\d{2}m-\d{2}d-\d{2}h-\d{2}m-\d{2}s"

    try:
        client = net.get_ssh_client(trossen_node.ip, trossen_node.user)
        # --- Update repo and venv ---
        update_cmd = (
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "git -C ~/tatbot pull && "
            "cd ~/tatbot/src/0.3 && "
            "deactivate >/dev/null 2>&1 || true && "
            "rm -rf .venv && "
            "rm -f uv.lock && "
            "uv venv && "
            f"uv pip install '{trossen_node.deps}'"
        )
        exit_code, out, err = net._run_remote_command(client, update_cmd, timeout=300)
        if exit_code != 0:
            client.close()
            return f"❌ Update failed on trossen-ai.\n{err or out}"

        # List scan directories before
        pre_cmd = f"ls -1 {remote_output_dir}"
        _, pre_out, _ = net._run_remote_command(client, pre_cmd)
        pre_dirs = set(re.findall(scan_dir_pattern, pre_out))

        # --- Run the scan ---
        scan_cmd = (
            f"export PATH=\"$HOME/.local/bin:$PATH\" && "
            f"cd ~/tatbot/src/0.3 && "
            f"[ -f .env ] && set -a && . .env && set +a; "
            f"uv run bot_scan.py"
        )
        exit_code, scan_out, scan_err = net._run_remote_command(client, scan_cmd, timeout=600)
        if exit_code != 0:
            client.close()
            return f"❌ Scan failed on trossen-ai.\n{scan_err or scan_out}"

        # List scan directories after
        _, post_out, _ = net._run_remote_command(client, pre_cmd)
        post_dirs = set(re.findall(scan_dir_pattern, post_out))
        client.close()

        new_dirs = post_dirs - pre_dirs
        if not new_dirs:
            return "❌ Could not find new scan output directory after running scan."
        # If multiple, pick the most recent (sorted by name, which is time-based)
        scan_dir = sorted(new_dirs)[-1]
        remote_scan_path = f"/home/{trossen_node.user}/tatbot/output/record/{scan_dir}"
        local_scan_path = os.path.expanduser(f"~/tatbot/output/record/{scan_dir}")

        # Step 2: Copy directory from trossen-ai to local node
        status = net.transfer_directory_from_node("trossen-ai", remote_scan_path, local_scan_path)
        if not status.startswith("✅"):
            return status

        # Step 3: Distribute to rpi1 and ook
        import tarfile
        tar_path = local_scan_path + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(local_scan_path, arcname=scan_dir)
        net.transfer_files_to_nodes(
            local_path=tar_path,
            remote_path=f"~/tatbot/output/record/{scan_dir}.tar.gz",
            node_names=["rpi1", "ook"],
            direction="put",
        )
        untar_cmd = f"mkdir -p ~/tatbot/output/record && tar -xzf ~/tatbot/output/record/{scan_dir}.tar.gz -C ~/tatbot/output/record"
        net.run_command_on_nodes(untar_cmd, node_names=["rpi1", "ook"])
        cleanup_cmd = f"rm -f ~/tatbot/output/record/{scan_dir}.tar.gz"
        net.run_command_on_nodes(cleanup_cmd, node_names=["rpi1", "ook"])
        try:
            os.remove(tar_path)
        except Exception:
            pass

        return f"✅ Scan complete. Output directory: {scan_dir}\nUpdated, scanned, and copied to local, rpi1, and ook."
    except Exception as e:
        log.error(f"scan_and_distribute failed: {e}")
        return f"❌ scan_and_distribute failed: {e}"

@mcp.tool(description="Performs a plan on trossen-ai using bot_plan.py, first copies the plan to trossen-ai, then runs the plan, then copies the resulting output directory to the local node and rpi1.")
def run_robot_plan(plan_name: str = "bench") -> str:
    """
    1. Finds the plan in output/plans/<plan_name>
    2. Copies the plan directory to trossen-ai
    3. Updates the tatbot repo and uv venv on trossen-ai.
    4. Runs `uv run bot_plan.py --plan_dir ~/tatbot/output/plans/<plan_name>` on trossen-ai.
    5. Finds the new plan recording output directory (plan-<plan_name>-<timestamp>).
    6. Copies the plan recording output directory from trossen-ai to the local node and rpi1.
    """
    trossen_node = next((n for n in net.nodes if n.name == "trossen-ai"), None)
    if not trossen_node:
        return "❌ trossen-ai node not found in configuration."
    if net.is_local_node(trossen_node):
        return "❌ trossen-ai is the local node; this function is intended for remote execution."

    local_plan_dir = os.path.expanduser(f"~/tatbot/output/plans/{plan_name}")
    remote_output_dir = "~/tatbot/output/record"
    plan_dir_pattern = rf"plan-{plan_name}-\\d{{4}}y-\\d{{2}}m-\\d{{2}}d-\\d{{2}}h-\\d{{2}}m-\\d{{2}}s"

    # Step 1: Copy the plan directory to trossen-ai (tar for transfer)
    tar_path = local_plan_dir + ".tar.gz"
    if not os.path.exists(local_plan_dir):
        return f"❌ Local plan directory not found: {local_plan_dir}"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_plan_dir, arcname=plan_name)
    net.transfer_files_to_nodes(
        local_path=tar_path,
        remote_path=f"~/tatbot/output/plans/{plan_name}.tar.gz",
        node_names=["trossen-ai"],
        direction="put",
    )
    # Untar on trossen-ai
    untar_cmd = f"mkdir -p ~/tatbot/output/plans && tar -xzf ~/tatbot/output/plans/{plan_name}.tar.gz -C ~/tatbot/output/plans"
    net.run_command_on_nodes(untar_cmd, node_names=["trossen-ai"])
    cleanup_cmd = f"rm -f ~/tatbot/output/plans/{plan_name}.tar.gz"
    net.run_command_on_nodes(cleanup_cmd, node_names=["trossen-ai"])
    try:
        os.remove(tar_path)
    except Exception:
        pass

    try:
        client = net.get_ssh_client(trossen_node.ip, trossen_node.user)
        # --- Update repo and venv ---
        update_cmd = (
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "git -C ~/tatbot pull && "
            "cd ~/tatbot/src/0.3 && "
            "deactivate >/dev/null 2>&1 || true && "
            "rm -rf .venv && "
            "rm -f uv.lock && "
            "uv venv && "
            f"uv pip install '{trossen_node.deps}'"
        )
        exit_code, out, err = net._run_remote_command(client, update_cmd, timeout=300)
        if exit_code != 0:
            client.close()
            return f"❌ Update failed on trossen-ai.\n{err or out}"

        # List plan output directories before
        pre_cmd = f"ls -1 {remote_output_dir}"
        _, pre_out, _ = net._run_remote_command(client, pre_cmd)
        pre_dirs = set(re.findall(plan_dir_pattern, pre_out))

        # --- Run the plan ---
        plan_cmd = (
            f"export PATH=\"$HOME/.local/bin:$PATH\" && "
            f"cd ~/tatbot/src/0.3 && "
            f"[ -f .env ] && set -a && . .env && set +a; "
            f"uv run bot_plan.py --plan_dir ~/tatbot/output/plans/{plan_name}"
        )
        exit_code, plan_out, plan_err = net._run_remote_command(client, plan_cmd, timeout=1200)
        if exit_code != 0:
            client.close()
            return f"❌ Plan execution failed on trossen-ai.\n{plan_err or plan_out}"

        # List plan output directories after
        _, post_out, _ = net._run_remote_command(client, pre_cmd)
        post_dirs = set(re.findall(plan_dir_pattern, post_out))
        client.close()

        new_dirs = post_dirs - pre_dirs
        if not new_dirs:
            return "❌ Could not find new plan output directory after running plan."
        plan_output_dir = sorted(new_dirs)[-1]
        remote_plan_output_path = f"/home/{trossen_node.user}/tatbot/output/record/{plan_output_dir}"
        local_plan_output_path = os.path.expanduser(f"~/tatbot/output/record/{plan_output_dir}")

        # Step 2: Copy directory from trossen-ai to local node
        status = net.transfer_directory_from_node("trossen-ai", remote_plan_output_path, local_plan_output_path)
        if not status.startswith("✅"):
            return status

        # Step 3: Distribute to rpi1
        tar_path = local_plan_output_path + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(local_plan_output_path, arcname=plan_output_dir)
        net.transfer_files_to_nodes(
            local_path=tar_path,
            remote_path=f"~/tatbot/output/record/{plan_output_dir}.tar.gz",
            node_names=["rpi1"],
            direction="put",
        )
        untar_cmd = f"mkdir -p ~/tatbot/output/record && tar -xzf ~/tatbot/output/record/{plan_output_dir}.tar.gz -C ~/tatbot/output/record"
        net.run_command_on_nodes(untar_cmd, node_names=["rpi1"])
        cleanup_cmd = f"rm -f ~/tatbot/output/record/{plan_output_dir}.tar.gz"
        net.run_command_on_nodes(cleanup_cmd, node_names=["rpi1"])
        try:
            os.remove(tar_path)
        except Exception:
            pass

        return f"✅ Plan complete. Output directory: {plan_output_dir}\nUpdated, ran plan, and copied to local and rpi1."
    except Exception as e:
        log.error(f"run_robot_plan failed: {e}")
        return f"❌ run_robot_plan failed: {e}"

def run_mcp(config: MCPConfig):
    log.info("🔌 Starting MCP server")
    mcp.run(transport=config.transport)

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    run_mcp(args)