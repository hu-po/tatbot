FROM nvcr.io/nvidia/l4t-ml:r36.3.0-py3

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    libjpeg-dev \
    libssl-dev \
    && apt-get clean

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip

# Install `uv` for fast env + dependency mgmt
RUN curl -Ls https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.cargo/bin:$PATH"

# Create project directory
WORKDIR /workspace/tatbot

# Copy project files (optional)
# COPY . /workspace/tatbot

# Install JAX with ARM64 + CUDA 12.6 compatible version
RUN uv pip install --upgrade --no-cache-dir \
    jax==0.4.20 \
    jaxlib==0.4.20+cuda12.cudnn89 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html