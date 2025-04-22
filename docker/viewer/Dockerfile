FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY pyproject.toml /app/
COPY docker/viewer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy viewer code
COPY viewer /app/viewer

# Create necessary directories for Three.js libraries
RUN mkdir -p /app/viewer/static/js/libs/examples/jsm/controls \
    /app/viewer/static/js/libs/examples/jsm/loaders \
    /app/viewer/static/js/libs/examples/jsm/libs

# Copy Three.js libraries
COPY viewer/static/js/libs/OrbitControls.js /app/viewer/static/js/libs/examples/jsm/controls/
COPY viewer/static/js/libs/TransformControls.js /app/viewer/static/js/libs/examples/jsm/controls/
COPY viewer/static/js/libs/USDZLoader.js /app/viewer/static/js/libs/examples/jsm/loaders/
COPY viewer/static/js/libs/three.module.js /app/viewer/static/js/libs/
COPY viewer/static/js/libs/fflate.module.js /app/viewer/static/js/libs/examples/jsm/libs/

# Create mount points for output, assets, and logos
RUN mkdir -p /app/output /app/assets /app/assets/logos

# Copy favicon
COPY assets/logos/favicon.ico /app/assets/logos/favicon.ico

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=viewer/server.py
ENV FLASK_ENV=development

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]