import os
from flask import Flask, jsonify, request, send_from_directory

# Assume the script is run from the project root or adjust path accordingly
# Or better, determine path relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output') # Path to your USD files
ASSETS_FOLDER = os.path.join(PROJECT_ROOT, 'assets') # Path to your assets
LOGOS_FOLDER = os.path.join(PROJECT_ROOT, 'assets/logos') # Path to logos

# Ensure the output folder exists
if not os.path.isdir(OUTPUT_FOLDER):
     print(f"Warning: Output folder {OUTPUT_FOLDER} not found.")
     # Optionally create it: os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Ensure the assets folder exists
if not os.path.isdir(ASSETS_FOLDER):
     print(f"Warning: Assets folder {ASSETS_FOLDER} not found.")

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')

# Route to serve the main viewer page
@app.route('/')
def index():
    # Read the HTML file
    with open(os.path.join(app.static_folder, 'index.html'), 'r') as f:
        html = f.read()
    
    # Inject the USD_FILE environment variable
    usd_file = os.environ.get('USD_FILE', 'stencil.usdz')
    html = html.replace('</head>', f'<script>window.USD_FILE = "{usd_file}";</script></head>')
    
    return html

# Route to serve favicon
@app.route('/favicon.ico')
def serve_favicon():
    return send_from_directory(LOGOS_FOLDER, 'favicon.ico')

# Route to serve USD files from the output directory
@app.route('/output/<path:filename>')
def serve_output_file(filename):
     if not os.path.exists(OUTPUT_FOLDER):
          return "Output directory not found", 404
     print(f"Attempting to serve: {filename} from {OUTPUT_FOLDER}")
     # Basic security check (optional but recommended)
     safe_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, filename))
     if not safe_path.startswith(OUTPUT_FOLDER):
          return "Forbidden", 403
     return send_from_directory(OUTPUT_FOLDER, filename)

# Route to serve USD files from the assets directory
@app.route('/assets/<path:filename>')
def serve_assets_file(filename):
     if not os.path.exists(ASSETS_FOLDER):
          return "Assets directory not found", 404
     print(f"Attempting to serve: {filename} from {ASSETS_FOLDER}")
     # Basic security check (optional but recommended)
     safe_path = os.path.abspath(os.path.join(ASSETS_FOLDER, filename))
     if not safe_path.startswith(ASSETS_FOLDER):
          return "Forbidden", 403
     return send_from_directory(ASSETS_FOLDER, filename)

# API Endpoint: Provide scene data (e.g., path to USD file)
@app.route('/api/scene_data')
def get_scene_data():
    # Get the filename from query parameter, default to scene1.usd if not provided
    filename = request.args.get('file', 'stencil.usdz')
    
    # Determine which folder to use based on the path
    if filename.startswith('output/'):
        filename = filename[7:]
        base_folder = OUTPUT_FOLDER
        url_prefix = '/output'
    elif filename.startswith('assets/'):
        filename = filename[7:]
        base_folder = ASSETS_FOLDER
        url_prefix = '/assets'
    else:
        # Default to output folder if no prefix
        base_folder = OUTPUT_FOLDER
        url_prefix = '/output'

    usd_file_path = os.path.join(base_folder, filename)

    if os.path.exists(usd_file_path):
        # Return the URL path the frontend can use
        return jsonify({'usd_url': f'{url_prefix}/{filename}'})
    else:
        print(f"Error: Scene file {usd_file_path} not found.")
        return jsonify({'error': f'Scene file {filename} not found in {base_folder}'}), 404

# API Endpoint: Receive updated object transforms from the frontend
@app.route('/api/update_transform', methods=['POST'])
def update_transform():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON payload'}), 400

    object_id = data.get('objectId')
    position = data.get('position')
    quaternion = data.get('quaternion')
    scale = data.get('scale')

    # --- TODO: Implement logic to save this data ---
    # Example: Update a config file, modify the USD file directly, etc.
    print(f"Received update for object: {object_id}")
    print(f"  Position: {position}")
    print(f"  Quaternion: {quaternion}")
    print(f"  Scale: {scale}")
    # ------------------------------------------------

    return jsonify({'status': 'success', 'objectId': object_id})

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible within Docker
    # Use a port like 8080
    app.run(host='0.0.0.0', port=8080, debug=True) # Set debug=False for production