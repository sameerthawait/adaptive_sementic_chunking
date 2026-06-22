#!/bin/bash
set -e

echo "=========================================================="
echo "    Adaptive Semantic Chunking (ASC) Setup Script"
echo "=========================================================="

# 1. Check Python >= 3.11
echo "Checking Python version..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "Error: Python is not installed. Please install Python 3.11+."
    exit 1
fi

$PYTHON_CMD -c "
import sys
major, minor = sys.version_info.major, sys.version_info.minor
if (major, minor) < (3, 11):
    print(f'Error: Python {major}.{minor} detected. ASC requires Python >= 3.11.')
    sys.exit(1)
else:
    print(f'Python version check passed: {major}.{minor}')
"

# 2. Create venv in ./venv
echo "Creating virtual environment in ./venv..."
if command -v python3.12 >/dev/null 2>&1; then
    echo "Using Python 3.12 to ensure pre-compiled binary wheels are available..."
    python3.12 -m venv venv
elif command -v python3.11 >/dev/null 2>&1; then
    echo "Using Python 3.11..."
    python3.11 -m venv venv
else
    echo "Falling back to default Python command..."
    $PYTHON_CMD -m venv venv
fi

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Warning: Could not find activation script. Attempting to use path direct execution."
fi

# 3. Install requirements
echo "Installing requirements (resolving httpx/ollama conflicts)..."
grep -v "^ollama==" requirements.txt > temp_requirements.txt

if command -v uv >/dev/null 2>&1; then
    echo "Found 'uv' package manager. Using uv for fast, cached installation..."
    uv pip install -r temp_requirements.txt --python venv
    rm temp_requirements.txt
    uv pip install ollama==0.4.4 --no-deps --python venv
else
    echo "Using pip for installation..."
    pip install --upgrade pip
    pip install -r temp_requirements.txt
    rm temp_requirements.txt
    pip install ollama==0.4.4 --no-deps
fi

# 4. Download NLTK datasets
echo "Downloading NLTK resources (punkt, punkt_tab)..."
python -m nltk.downloader punkt punkt_tab

# 5. Pull Ollama models
echo "Pulling local Ollama models (llama3.2:3b, nomic-embed-text)..."
if command -v ollama >/dev/null 2>&1; then
    echo "Pulling llama3.2:3b..."
    ollama pull llama3.2:3b || echo "Warning: Failed to pull llama3.2:3b. Make sure Ollama is running."
    echo "Pulling nomic-embed-text..."
    ollama pull nomic-embed-text || echo "Warning: Failed to pull nomic-embed-text. Make sure Ollama is running."
else
    echo "Warning: 'ollama' CLI not found. Please ensure Ollama is installed and running."
fi

# 6. Create ./data/chromadb directory
echo "Creating database directory ./data/chromadb..."
mkdir -p ./data/chromadb

echo "=========================================================="
echo "    Setup Completed Successfully!"
echo "=========================================================="
echo "Next Steps:"
echo "1. Activate the environment:"
echo "   source venv/bin/activate    (Linux/macOS)"
echo "   venv\\Scripts\\activate      (Windows)"
echo "2. Copy the .env.example file to .env and configure if needed:"
echo "   cp .env.example .env"
echo "3. Ensure Ollama is running local server."
echo "4. Run benchmarks or API server:"
echo "   python main.py benchmark"
echo "   python main.py serve"
echo "=========================================================="
