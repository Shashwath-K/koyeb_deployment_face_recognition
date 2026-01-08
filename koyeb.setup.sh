#!/bin/bash
# koyeb-setup.sh - Setup script for Koyeb deployment with PyTorch

echo "🚀 Setting up Koyeb deployment with PyTorch support..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Create requirements.txt with PyTorch CPU version
echo "📄 Creating requirements.txt with PyTorch CPU..."
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
EOF

# Create necessary directories
mkdir -p static/uploads
mkdir -p models

# Create .dockerignore to reduce image size
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

# Create minimal app.py if not exists
if [ ! -f "app.py" ]; then
    echo "📄 Creating minimal app.py..."
    cat > app.py << 'EOF'
from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        import torch
        import cv2
        import numpy as np
        from facenet_pytorch import InceptionResnetV1
        
        return jsonify({
            'status': 'healthy',
            'message': 'Face Attendance API with PyTorch is running',
            'torch_version': torch.__version__,
            'cv2_version': cv2.__version__,
            'numpy_version': np.__version__
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def index():
    return jsonify({
        'name': 'Face Attendance API',
        'version': '1.0.0',
        'framework': 'Flask + PyTorch',
        'endpoints': ['/', '/api/health']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
EOF
fi

# Create Docker build helper script
echo "📄 Creating build-docker.sh..."
cat > build-docker.sh << 'EOF'
#!/bin/bash
echo "🔨 Building optimized Docker image with PyTorch CPU..."
echo "Note: This will take a few minutes due to PyTorch size"
echo ""

# Build with cache
docker build --progress=plain -t face-attendance-pytorch:latest .

echo ""
echo "✅ Build complete!"
echo ""
echo "📊 Image size information:"
docker images face-attendance-pytorch:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo ""
echo "🚀 To run locally:"
echo "  docker run -p 8000:8000 -e PORT=8000 face-attendance-pytorch:latest"
echo ""
echo "📤 To push to Docker Hub:"
echo "  docker tag face-attendance-pytorch:latest YOUR_USERNAME/face-attendance-pytorch:latest"
echo "  docker push YOUR_USERNAME/face-attendance-pytorch:latest"
EOF

chmod +x build-docker.sh

# Set execute permissions
chmod +x koyeb-setup.sh

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Build the image: ./build-docker.sh"
echo "2. Test locally: docker run -p 8000:8000 -e PORT=8000 face-attendance-pytorch:latest"
echo "3. Visit: http://localhost:8000/api/health"
echo "4. Push to Docker Hub"
echo "5. Deploy on Koyeb using koyeb.yaml"
echo ""
echo "⚠️  Important:"
echo "- Image size will be ~2.5-3GB due to PyTorch"
echo "- Use 'Large' or higher instance type in Koyeb (minimum 8GB disk)"
echo "- Build may take 10-15 minutes"