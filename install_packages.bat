@echo off
echo 🔄 Installing PDF Vectorization Pipeline (Simple Installation)...

REM Install packages individually to avoid conflicts
echo 📦 Installing core packages...
python -m pip install --only-binary=all azure-storage-blob python-dotenv tqdm psutil

echo 📦 Installing PDF processing...
python -m pip install --only-binary=all PyPDF2 pdfplumber tiktoken

echo 📦 Installing numpy and pandas...
python -m pip install --only-binary=all numpy pandas

echo 📦 Installing PyTorch (CPU)...
python -m pip install --only-binary=all torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo 📦 Installing sentence transformers...
python -m pip install --only-binary=all sentence-transformers

echo 📦 Installing ChromaDB...
python -m pip install --only-binary=all chromadb

echo ✅ Installation complete! Run 'python validate_environment.py' to verify.
pause
