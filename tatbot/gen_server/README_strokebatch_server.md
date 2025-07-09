# Strokebatch Server

This directory contains scripts for running the strokebatch computation as a separate server on `ojo`, allowing the robot `trossen-ai` to offload the computationally intensive strokebatch generation.

## Architecture

- **Server** (`strokebatch_server.py`): Runs on `ojo` and handles strokebatch computation requests
- **Client** (`strokebatch_client.py`): Used by the robot to request strokebatch computation
- **Dependencies**: Separated so the robot only needs minimal dependencies

## Setup on ojo (Server)

1. **Install server dependencies:**
   ```bash
   cd /home/oop/tatbot
   pip install -r scripts/requirements_server.txt
   ```

2. **Test the server:**
   ```bash
   python scripts/strokebatch_server.py --debug
   ```

3. **Install as systemd service (optional):**
   ```bash
   sudo cp scripts/strokebatch-server.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable strokebatch-server
   sudo systemctl start strokebatch-server
   ```

4. **Check service status:**
   ```bash
   sudo systemctl status strokebatch-server
   ```

## Setup on trossen-ai (Robot)

1. **Install client dependencies:**
   ```bash
   pip install -r scripts/requirements_client.txt
   ```

2. **Test connection:**
   ```bash
   python scripts/strokebatch_client.py --health
   ```

## Usage

### From the robot:

1. **Check server health:**
   ```bash
   python scripts/strokebatch_client.py --health
   ```

2. **Compute strokebatch:**
   ```bash
   python scripts/strokebatch_client.py scene.yaml strokelist.yaml --output strokebatch.safetensors
   ```

3. **From Python code:**
   ```python
   from scripts.strokebatch_client import call_strokebatch_server
   
   result_path = call_strokebatch_server(
       scene_path="scene.yaml",
       strokelist_path="strokelist.yaml",
       server_url="http://ojo:5000",
       output_path="strokebatch.safetensors"
   )
   ```

### Integration with record script:

Modify the record script to use the server instead of local computation:

```python
# Replace this line in record.py:
# strokebatch: StrokeBatch = strokebatch_from_strokes(scene=scene, strokelist=strokes)

# With this:
from scripts.strokebatch_client import call_strokebatch_server
import tempfile

# Save scene and strokelist to temp files
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    scene.to_yaml(f.name)
    scene_path = f.name

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    strokelist.to_yaml(f.name)
    strokelist_path = f.name

# Call server
strokebatch_path = call_strokebatch_server(scene_path, strokelist_path)
strokebatch = StrokeBatch.load(strokebatch_path)
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "strokebatch_server",
  "version": "1.0.0"
}
```

### POST /compute_strokebatch
Compute strokebatch from scene and strokelist files.

**Request:** Multipart form data with:
- `scene`: YAML file containing scene configuration
- `strokelist`: YAML file containing stroke list

**Response:**
- `200`: safetensors file containing the computed strokebatch
- `400`: Bad request (missing files, invalid data)
- `500`: Server error during computation

## Troubleshooting

1. **Server won't start:**
   - Check if Flask is installed: `pip install flask`
   - Check if tatbot is in PYTHONPATH
   - Check logs: `sudo journalctl -u strokebatch-server`

2. **Client can't connect:**
   - Check if server is running: `curl http://ojo:5000/health`
   - Check network connectivity: `ping ojo`
   - Check firewall settings

3. **Computation fails:**
   - Check server logs for detailed error messages
   - Verify input files are valid YAML
   - Check if all tatbot dependencies are available on server

## Dependencies

### Server (ojo):
- `flask` - Web framework
- `werkzeug` - WSGI utilities
- `tatbot[gen]` - Core tatbot with generation dependencies

### Client (trossen-ai):
- `requests` - HTTP client
- `tatbot[bot]` - Core tatbot with robot dependencies 