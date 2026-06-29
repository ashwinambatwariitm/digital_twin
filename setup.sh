#!/bin/bash
# ─────────────────────────────────────────────────────────────
# setup.sh
# One-time setup for GraphCast India on RTX 3050 laptop
# ─────────────────────────────────────────────────────────────

set -e
echo "======================================================"
echo " GraphCast India — Setup"
echo " PS-5 Bharatiya Antariksh Hackathon 2026"
echo "======================================================"

# ── Check Python ─────────────────────────────────────────────
echo ""
echo "[1/6] Checking Python..."
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# ── Check CUDA ───────────────────────────────────────────────
echo ""
echo "[2/6] Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "✅ CUDA found"
else
    echo "⚠️  CUDA not found — will use CPU (much slower)"
fi

# ── Create virtual environment ───────────────────────────────
echo ""
echo "[3/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ venv created"
else
    echo "   venv already exists"
fi

source venv/bin/activate

# ── Install PyTorch with CUDA ─────────────────────────────────
echo ""
echo "[4/6] Installing PyTorch (CUDA 11.8)..."
pip install --upgrade pip --quiet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
echo "✅ PyTorch installed"

# ── Install PyTorch Geometric ─────────────────────────────────
echo ""
echo "[5/6] Installing PyTorch Geometric..."
pip install torch-geometric --quiet
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html --quiet
echo "✅ PyTorch Geometric installed"

# ── Install remaining requirements ───────────────────────────
echo ""
echo "[6/6] Installing remaining packages..."
pip install -r requirements.txt --quiet
echo "✅ All packages installed"

# ── Create output directories ─────────────────────────────────
mkdir -p data/raw/imd
mkdir -p data/raw/insat
mkdir -p data/processed
mkdir -p training/checkpoints
mkdir -p evaluation/results
mkdir -p visualization/figures

echo ""
echo "======================================================"
echo " ✅ Setup complete!"
echo ""
echo " Quick start (synthetic data — no downloads needed):"
echo "   source venv/bin/activate"
echo "   python run.py --mode all"
echo ""
echo " With real data:"
echo "   1. Download IMD data to data/raw/imd/"
echo "      https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html"
echo "   2. Download INSAT data to data/raw/insat/"
echo "      https://www.mosdac.gov.in"
echo "   3. python run.py --mode all --real_data --year 2023"
echo "======================================================"