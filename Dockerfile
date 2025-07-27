FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    git \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install numpy explicitly to avoid build conflicts
RUN pip install --no-cache-dir "numpy>=1.21.0,<1.25.0"

# Install detectron2 from source (more reliable than pre-built wheels)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure the local LightGBM model files are copied and accessible
# These are critical for Round 1A processing within Round 1B
COPY models/token_type_lightgbm.model /app/models/
COPY models/paragraph_extraction_lightgbm.model /app/models/
COPY models/config.json /app/models/

# Create missing __init__.py files for all Python packages (this handles all directories safely)
RUN find /app/src -type d -exec touch {}/__init__.py \;

# Install the package in development mode to handle dependencies
RUN pip install -e .

# Download HuggingFace models during build time so they're available offline
RUN python -c "\
import os; \
os.environ['PYTHONPATH'] = '/app:/app/src'; \
from huggingface_hub import hf_hub_download; \
print('Downloading pdf_tokens_type.model...'); \
pdf_tokens_type_model = hf_hub_download( \
    repo_id='HURIDOCS/pdf-segmentation', \
    filename='pdf_tokens_type.model', \
    revision='c71f833500707201db9f3649a6d2010d3ce9d4c9', \
    cache_dir='/app/models/.huggingface' \
); \
print(f'Downloaded to: {pdf_tokens_type_model}'); \
print('Downloading token type finding config...'); \
token_type_finding_config_path = hf_hub_download( \
    repo_id='HURIDOCS/pdf-segmentation', \
    filename='tag_type_finding_model_config.txt', \
    revision='7d98776dd34acb2fe3a06495c82e64b9c84bdc16', \
    cache_dir='/app/models/.huggingface' \
); \
print(f'Downloaded to: {token_type_finding_config_path}'); \
print('All models downloaded successfully!') \
"

# Download Round 1B models for offline execution
RUN python -c "\
import os; \
os.environ['PYTHONPATH'] = '/app:/app/src'; \
os.environ['CUDA_VISIBLE_DEVICES'] = ''; \
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'; \
print('Downloading Round 1B models for offline execution...'); \
print('1. Downloading sentence-transformers model...'); \
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models'); \
print('   ✅ all-MiniLM-L6-v2 model downloaded'); \
print('2. Downloading summarizer model...'); \
from summarizer import Summarizer; \
summarizer = Summarizer('distilbert-base-uncased'); \
print('   ✅ distilbert-base-uncased model downloaded'); \
print('🎉 All Round 1B models downloaded successfully!') \
"

# Ensure models directory exists and is accessible
RUN mkdir -p /app/models && chmod -R 755 /app/models

# Ensure xmls directory exists and is accessible
RUN mkdir -p /app/xmls && chmod -R 755 /app/xmls

# Verify that the local LightGBM model files are present and accessible
RUN echo "Checking for LightGBM model files..." && \
    ls -la /app/models/ && \
    if [ ! -f /app/models/token_type_lightgbm.model ]; then \
        echo "ERROR: token_type_lightgbm.model not found!"; \
        exit 1; \
    fi && \
    if [ ! -f /app/models/paragraph_extraction_lightgbm.model ]; then \
        echo "ERROR: paragraph_extraction_lightgbm.model not found!"; \
        exit 1; \
    fi && \
    echo "✅ LightGBM models verified successfully!"

# Create a symlink for model configurations if they don't exist
RUN if [ ! -f /app/models/config.json ]; then echo '{"tracking": true}' > /app/models/config.json; fi

# Set environment variables for CPU-only execution and proper module resolution
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL="2"
ENV PYTHONPATH="/app:/app/src"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Make entrypoint scripts executable
RUN chmod +x docker_entrypoint.py round1b_docker_entrypoint.py

# Set Round 1B as the default entrypoint for this solution
# Can be overridden if needed for Round 1A compatibility
ENTRYPOINT ["python", "round1b_docker_entrypoint.py"] 