import concurrent.futures
import json
import logging
import os
import re
import tarfile
from dataclasses import dataclass
from typing import List, Optional

import psutil
from mcp.server.fastmcp import FastMCP

from tatbot.net.net import NetworkManager
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('run_mcp', '🔌')

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

@mcp.tool(description="Ping nodes and report connectivity status.")
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

@mcp.tool(description="Update tatbot repo and Python env on nodes via git pull and uv.")
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
                "cd ~/tatbot/src && "
                "deactivate >/dev/null 2>&1 || true && "
                "rm -rf .venv && "
                "rm -f uv.lock && "
                "uv venv --prompt=\"tatbot\" && "
                f"uv pip install {node.deps}"
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

@mcp.tool(description="Report CPU and RAM usage for nodes.")
def node_cpu_ram_usage(nodes: Optional[List[str]] = None) -> dict[str, dict]:
    log.info(f"🔌 Getting CPU/RAM usage for nodes: {nodes or 'all'}")
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
            "cd ~/tatbot/src && "
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

@mcp.tool(description="Power off remote nodes (requires passwordless sudo). Skips local node.")
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

@mcp.tool(description="Start visualization on rpi1: update, kill old procs, launch viz, open Chromium.")
def run_viz(
    viz_type: str,  # 'test', 'plan', or 'scan'
    name: str,       # plan name or scan name
) -> str:
    log.info(f"🔌 Running {viz_type} visualization on rpi1")

    rpi1_node = next((n for n in net.nodes if n.name == "rpi1"), None)
    if not rpi1_node:
        return "❌ rpi1 node not found in configuration."

    if net.is_local_node(rpi1_node):
        return "❌ rpi1 is the local node; this function is intended for remote execution."

    # Use update_nodes for environment update
    update_status = update_nodes(["rpi1"])
    if not update_status or "Failed" in update_status or "Exception" in update_status:
        return f"❌ Failed to update environment on rpi1.\n{update_status}"
    
    if viz_type == "test":
        viz_command = "_viz.py"
    elif viz_type == "plan":
        viz_command = f"uv run viz_plan.py --plan_dir ~/tatbot/output/plans/{name}"
    elif viz_type == "scan":
        viz_command = f"uv run viz_scan.py --scan_dir ~/tatbot/output/record/{name}"
    else:
        return f"❌ Invalid viz_type: {viz_type}"

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
            "echo exporting display... >> ~/chromium-viz.log\n"
            "export DISPLAY=:0\n"
            "export XAUTHORITY=/home/rpi1/.Xauthority\n"
            "echo launching viz process... >> ~/chromium-viz.log\n"
            "cd ~/tatbot/src\n"
            "source .venv/bin/activate\n"
            f"setsid uv run {viz_command} >> ~/chromium-viz.log 2>&1 &\n"
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

@mcp.tool(description="Run scan on trossen-ai, copy output to local and rpi1.")
def run_robot_scan() -> str:
    log.info(f"🔌 Running scan on trossen-ai")

    trossen_node = next((n for n in net.nodes if n.name == "trossen-ai"), None)
    if not trossen_node:
        return "❌ trossen-ai node not found in configuration."
    if net.is_local_node(trossen_node):
        return "❌ trossen-ai is the local node; this function is intended for remote execution."

    remote_output_dir = "~/tatbot/output/record"
    scan_dir_pattern = r"scan-\d{4}y-\d{2}m-\d{2}d-\d{2}h-\d{2}m-\d{2}s"

    # Use update_nodes for environment update
    update_status = update_nodes(["trossen-ai"])
    if not update_status or "Failed" in update_status or "Exception" in update_status:
        return f"❌ Failed to update environment on trossen-ai.\n{update_status}"

    try:
        client = net.get_ssh_client(trossen_node.ip, trossen_node.user)
        # List scan directories before
        pre_cmd = f"ls -1 {remote_output_dir}"
        _, pre_out, _ = net._run_remote_command(client, pre_cmd)
        pre_dirs = set(re.findall(scan_dir_pattern, pre_out))

        # --- Run the scan ---
        scan_cmd = (
            f"export PATH=\"$HOME/.local/bin:$PATH\" && "
            "cd ~/tatbot/src && "
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

        # Step 3: Distribute to rpi1 using the new tool
        dist_status = copy_data_between_nodes(
            data_type="recording",
            name=scan_dir,
            source_node="trossen-ai",
            destination_nodes=["rpi1"],
        )
        if not dist_status.startswith("✅"):
            return dist_status

        return f"✅ Scan complete. Output directory: {scan_dir}\nUpdated, scanned, and copied to local and rpi1."
    except Exception as e:
        log.error(f"scan_and_distribute failed: {e}")
        return f"❌ scan_and_distribute failed: {e}"

@mcp.tool(description="Run plan on trossen-ai, copy output to local and rpi1.")
def run_robot_plan(plan_name: str = "bench") -> str:
    log.info(f"🔌 Running plan on trossen-ai")

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
    log.info(f"[MCP] Local plan dir: {local_plan_dir}")
    log.info(f"[MCP] Tarball path: {tar_path}")
    if not os.path.exists(local_plan_dir):
        log.error(f"[MCP] Local plan directory does not exist: {local_plan_dir}")
        return f"❌ Local plan directory not found: {local_plan_dir}"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_plan_dir, arcname=plan_name)
    if not os.path.exists(tar_path):
        log.error(f"[MCP] Tarball was not created: {tar_path}")
        return f"❌ Tarball was not created: {tar_path}"
    tar_size = os.path.getsize(tar_path)
    log.info(f"[MCP] Tarball exists: {tar_path}, size: {tar_size} bytes")
    remote_tar_path = f"~/tatbot/output/plans/{plan_name}.tar.gz"
    log.info(f"[MCP] Remote tar path: {remote_tar_path}")
    net.transfer_files_to_nodes(
        local_path=tar_path,
        remote_path=remote_tar_path,
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

    # Use update_nodes for environment update
    update_status = update_nodes(["trossen-ai"])
    if not update_status or "Failed" in update_status or "Exception" in update_status:
        return f"❌ Failed to update environment on trossen-ai.\n{update_status}"

    try:
        client = net.get_ssh_client(trossen_node.ip, trossen_node.user)
        # List plan output directories before
        pre_cmd = f"ls -1 {remote_output_dir}"
        _, pre_out, _ = net._run_remote_command(client, pre_cmd)
        pre_dirs = set(re.findall(plan_dir_pattern, pre_out))

        # --- Run the plan ---
        plan_cmd = (
            f"export PATH=\"$HOME/.local/bin:$PATH\" && "
            f"cd ~/tatbot/src/{TATBOT_VERSION} && "
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
        log.info(f"[MCP] transfer_directory_from_node status: {status}")
        if not status.startswith("✅"):
            return status

        # Step 3: Distribute to rpi1 using the new tool
        dist_status = copy_data_between_nodes(
            data_type="recording",
            name=plan_output_dir,
            source_node="trossen-ai",
            destination_nodes=["rpi1"],
        )
        if not dist_status.startswith("✅"):
            return dist_status

        return f"✅ Plan complete. Output directory: {plan_output_dir}\nUpdated, ran plan, and copied to local and rpi1."
    except Exception as e:
        log.error(f"run_robot_plan failed: {e}")
        return f"❌ run_robot_plan failed: {e}"

@mcp.tool(description="Copy plan or recording dir from one node to others (tar/untar).")
def copy_data_between_nodes(
    data_type: str,  # 'plan' or 'recording'
    name: str,       # plan name or recording dir
    source_node: str,
    destination_nodes: List[str],
) -> str:
    log.info(f"🔌 Copying {data_type} from {source_node} to {destination_nodes}")
    
    if data_type not in ("plan", "recording"):
        return "❌ data_type must be 'plan' or 'recording'"
    if not destination_nodes:
        return "❌ No destination nodes specified."

    # Determine remote and local paths
    if data_type == "plan":
        remote_dir = f"~/tatbot/output/plans/{name}"
        local_dir = os.path.expanduser(f"~/tatbot/output/plans/{name}")
        remote_tar = f"~/tatbot/output/plans/{name}.tar.gz"
        local_tar = local_dir + ".tar.gz"
        remote_untar_dir = "~/tatbot/output/plans"
    else:
        remote_dir = f"~/tatbot/output/record/{name}"
        local_dir = os.path.expanduser(f"~/tatbot/output/record/{name}")
        remote_tar = f"~/tatbot/output/record/{name}.tar.gz"
        local_tar = local_dir + ".tar.gz"
        remote_untar_dir = "~/tatbot/output/record"

    # Step 1: Copy directory from source_node to local
    status = net.transfer_directory_from_node(source_node, remote_dir, local_dir)
    if not status.startswith("✅"):
        return status

    # Step 2: Tar the directory locally
    try:
        with tarfile.open(local_tar, "w:gz") as tar:
            tar.add(local_dir, arcname=name)
    except Exception as e:
        return f"❌ Failed to create tarball: {e}"

    # Step 3: Transfer tarball to destination nodes
    net.transfer_files_to_nodes(
        local_path=local_tar,
        remote_path=remote_tar,
        node_names=destination_nodes,
        direction="put",
    )

    # Step 4: Untar on each destination node
    untar_cmd = f"mkdir -p {remote_untar_dir} && tar -xzf {remote_tar} -C {remote_untar_dir}"
    net.run_command_on_nodes(untar_cmd, node_names=destination_nodes)
    cleanup_cmd = f"rm -f {remote_tar}"
    net.run_command_on_nodes(cleanup_cmd, node_names=destination_nodes)

    # Step 5: Clean up local tarball
    try:
        os.remove(local_tar)
    except Exception:
        pass

    return f"✅ {data_type.capitalize()} '{name}' copied from {source_node} to {', '.join(destination_nodes)}."

def run_mcp(config: MCPConfig):
    log.info("🔌 Starting MCP server")
    mcp.run(transport=config.transport)

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    run_mcp(args)