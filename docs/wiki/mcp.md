# MCP (Model Context Protocol)

- individual nodes run their own mcp servers, see `tatbot/.cursor/mcp.json`
- [`mcp-python`](https://github.com/modelcontextprotocol/python-sdk)

start all the mcp servers:

```bash
./scripts/update-nodes.sh
./scripts/start-mcp-servers.sh
```

mcp logs are written to `~/tatbot/nfs/mcp-logs/`

## Cursor

- Cursor IDE (on `ook`) is used to interact with mcp servers, to add an mcp server to cursor:
- `ctrl-shift-p` > "View: OpenMCP Settings" > Toggle the MCP server
- If you change the tools or restart the server, you must re-toggle the server above
- Sometimes exiting and re-opening cursor will fix "Bad Request" errors

## Servers

`base`
- generally runs on `ook`
- resource: docs

`ojo`
- start vla policy server
- start vggt model server
- kill all docker containers

`rpi1`
- 

`rpi2`
- 

`trossen-ai`
- perform a scan
- visualize a strokelist
- ping cameras
- run camera calibration