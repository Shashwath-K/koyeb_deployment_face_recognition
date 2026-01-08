#!/bin/bash
# koyeb-setup.sh - Setup script for Koyeb deployment with Docker

echo "🚀 Setting up Koyeb deployment with Docker..."

# Create optimized requirements.txt
echo "📄 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
flask==2.3.3
opencv-python-headless==4.8.1.78
numpy==1.24.3
torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
facenet-pytorch==2.5.3
pillow==10.1.0
psycopg2-binary==2.9.7
gdown==4.7.1
gunicorn==21.2.0
flask-cors==4.0.0
python-dotenv==1.0.0
setuptools==68.2.2  # Added to fix the build error
EOF

# Create .python-version to specify Python 3.10
echo "📄 Creating .python-version..."
echo "3.10.13" > .python-version

# Create runtime.txt for Heroku buildpack compatibility
echo "📄 Creating runtime.txt..."
echo "python-3.10.13" > runtime.txt

# Create .dockerignore
echo "📄 Creating .dockerignore..."
cat > .dockerignore << 'EOF'
.git
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
.env
.gitignore
README.md
*.log
*.tmp
.DS_Store
.vscode
.idea
*.db
*.sqlite3
.ipynb_checkpoints
notebooks/
docs/
tests/
test/
data/
dataset/
samples/
examples/
*.jpg
*.jpeg
*.png
*.mp4
*.avi
*.zip
*.tar.gz
*.gz
*.7z
*.rar
EOF

# Create necessary directories
mkdir -p static/uploads
mkdir -p models

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Commit these files to GitHub"
echo "2. Go to Koyeb dashboard"
echo "3. Create App → Deploy from GitHub"
echo "4. Select your repository"
echo "5. Set Build Method to 'Dockerfile'"
echo "6. Set Dockerfile path to './Dockerfile'"
echo "7. Deploy!"