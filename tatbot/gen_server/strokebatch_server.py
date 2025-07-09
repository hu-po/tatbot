#!/usr/bin/env python3
"""
Strokebatch computation server for tatbot.

This server runs on ojo and waits for strokebatch computation requests from the robot.
It only depends on the core tatbot dependencies needed for strokebatch computation.

Usage:
    python scripts/strokebatch_server.py [--host HOST] [--port PORT]
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Optional

# Add tatbot to path
tatbot_root = Path(__file__).parent.parent
sys.path.insert(0, str(tatbot_root))

try:
    from flask import Flask, request, send_file, jsonify
    from werkzeug.utils import secure_filename
except ImportError:
    print("❌ Flask not installed. Install with: pip install flask")
    sys.exit(1)

try:
    from tatbot.data.scene import Scene
    from tatbot.data.stroke import StrokeList
    from tatbot.gen.batch import strokebatch_from_strokes
    from tatbot.utils.log import get_logger
except ImportError as e:
    print(f"❌ Tatbot dependencies not available: {e}")
    print("Make sure you're in the tatbot environment and tatbot is installed")
    sys.exit(1)

# Setup logging
log = get_logger('strokebatch_server', '🖥️')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure logging for Flask
logging.getLogger('werkzeug').setLevel(logging.INFO)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'strokebatch_server',
        'version': '1.0.0'
    })

@app.route('/compute_strokebatch', methods=['POST'])
def compute_strokebatch():
    """
    Compute strokebatch from scene and strokelist files.
    
    Expects multipart form data with:
    - scene: YAML file containing scene configuration
    - strokelist: YAML file containing stroke list
    
    Returns:
    - 200: safetensors file containing the computed strokebatch
    - 400: Bad request (missing files, invalid data)
    - 500: Server error during computation
    """
    try:
        # Check if files are present
        if 'scene' not in request.files:
            return jsonify({'error': 'Missing scene file'}), 400
        if 'strokelist' not in request.files:
            return jsonify({'error': 'Missing strokelist file'}), 400
        
        scene_file = request.files['scene']
        strokelist_file = request.files['strokelist']
        
        # Check if files are empty
        if scene_file.filename == '':
            return jsonify({'error': 'Scene file is empty'}), 400
        if strokelist_file.filename == '':
            return jsonify({'error': 'Strokelist file is empty'}), 400
        
        log.info(f"📥 Received request: scene={scene_file.filename}, strokelist={strokelist_file.filename}")
        
        # Save uploaded files to temporary locations
        scene_path = None
        strokelist_path = None
        result_path = None
        
        try:
            # Save scene file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml', mode='wb') as f:
                scene_file.save(f.name)
                scene_path = f.name
            
            # Save strokelist file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml', mode='wb') as f:
                strokelist_file.save(f.name)
                strokelist_path = f.name
            
            log.info("📁 Saved uploaded files to temporary locations")
            
            # Load scene and strokelist
            log.info("🔄 Loading scene and strokelist...")
            scene = Scene.from_yaml(scene_path)
            strokelist = StrokeList.from_yaml(strokelist_path)
            
            log.info(f"📊 Scene loaded: {len(strokelist.strokes)} strokes")
            
            # Compute strokebatch
            log.info("🧮 Computing strokebatch...")
            strokebatch = strokebatch_from_strokes(scene=scene, strokelist=strokelist)
            
            log.info("✅ Strokebatch computation completed")
            
            # Save result to temporary file
            result_path = tempfile.mktemp(suffix='.safetensors')
            strokebatch.save(result_path)
            
            log.info(f"💾 Saved strokebatch to {result_path}")
            
            # Return the file
            return send_file(
                result_path,
                as_attachment=True,
                download_name='strokebatch.safetensors',
                mimetype='application/octet-stream'
            )
            
        finally:
            # Clean up temporary files
            for path in [scene_path, strokelist_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                        log.debug(f"🗑️ Cleaned up temporary file: {path}")
                    except OSError as e:
                        log.warning(f"⚠️ Failed to clean up {path}: {e}")
            
            # Note: result_path is cleaned up by Flask after sending
    
    except Exception as e:
        log.error(f"❌ Error processing request: {e}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large (max 50MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Strokebatch computation server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    log.info(f"🚀 Starting strokebatch server on {args.host}:{args.port}")
    log.info("📋 Available endpoints:")
    log.info("  GET  /health - Health check")
    log.info("  POST /compute_strokebatch - Compute strokebatch from scene and strokelist")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == '__main__':
    main() 