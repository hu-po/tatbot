# MCP (Model Context Protocol)

- individual nodes run their own mcp servers, see `tatbot/.cursor/mcp.json`
- [`mcp-python`](https://github.com/modelcontextprotocol/python-sdk)

## Cursor

- Cursor IDE (on `ook`) is used to interact with mcp servers, to add an mcp server to cursor:
- `ctrl-shift-p` > "View: OpenMCP Settings" > Toggle the MCP server
- If you change the tools or restart the server, you must re-toggle the server above

## Servers

`base`
- generally runs on `ook`
- resource: docs

`rpi1`
- 

`rpi2`
- 

`tai` (`trossen-ai`):
- perform a scan
- visualize a strokelist
- ping cameras
- run camera calibration

`ojo`
- start vla policy server
- start vggt model server
- kill all docker containers