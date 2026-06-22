Write-Host "==========================================================" -ForegroundColor Green
Write-Host "    Adaptive Semantic Chunking (ASC) Windows Setup Script" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# 1. Check Python version >= 3.11
Write-Host "Checking Python version..."
try {
    $pythonVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}
catch {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.11+."
    exit 1
}

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Detected Python version $pythonVer. ASC requires Python >= 3.11."
    exit 1
}
Write-Host "Python version check passed: $pythonVer"

# 2. Create venv in ./venv
Write-Host "Creating virtual environment in ./venv..."
if (py -0 | Select-String "3.12") {
    Write-Host "Using Python 3.12 to ensure pre-compiled binary wheels are available..."
    py -3.12 -m venv venv
}
elseif (py -0 | Select-String "3.11") {
    Write-Host "Using Python 3.11..."
    py -3.11 -m venv venv
}
else {
    Write-Host "Falling back to default Python..."
    python -m venv venv
}

# 3. Install requirements
Write-Host "Installing requirements (resolving httpx/ollama version conflicts)..."
Get-Content requirements.txt | Where-Object { $_ -notmatch "^ollama==" } | Set-Content temp_requirements.txt

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Found 'uv' package manager. Using uv for fast, cached installation..."
    & uv pip install -r temp_requirements.txt --python venv
    Remove-Item temp_requirements.txt
    & uv pip install ollama==0.4.4 --no-deps --python venv
}
else {
    Write-Host "Using pip for installation..."
    & venv\Scripts\pip install --upgrade pip
    & venv\Scripts\pip install -r temp_requirements.txt
    Remove-Item temp_requirements.txt
    & venv\Scripts\pip install ollama==0.4.4 --no-deps
}

# 4. Download NLTK datasets
Write-Host "Downloading NLTK resources (punkt, punkt_tab)..."
& venv\Scripts\python -m nltk.downloader punkt punkt_tab

# 5. Pull Ollama models
Write-Host "Pulling local Ollama models (llama3.2:3b, nomic-embed-text)..."
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Pulling llama3.2:3b..."
    ollama pull llama3.2:3b
    Write-Host "Pulling nomic-embed-text..."
    ollama pull nomic-embed-text
}
else {
    Write-Warning "ollama CLI not found. Please ensure Ollama is installed and running."
}

# 6. Create ./data/chromadb directory
Write-Host "Creating database directory ./data/chromadb..."
New-Item -ItemType Directory -Force -Path ./data/chromadb | Out-Null

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "    Setup Completed Successfully!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Next Steps:"
Write-Host "1. Activate the environment:"
Write-Host "   .\\venv\\Scripts\\Activate.ps1"
Write-Host "2. Copy .env.example to .env:"
Write-Host "   Copy-Item .env.example .env"
Write-Host "3. Ensure Ollama is running."
Write-Host "4. Run benchmarks or API server:"
Write-Host "   python main.py --benchmark"
Write-Host "   python main.py --server"
