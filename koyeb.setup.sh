#!/bin/bash
# koyeb-setup.sh

echo "🚀 Setting up Koyeb deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "📄 Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
Flask==2.3.3
Flask-CORS==4.0.0
numpy==1.24.3
opencv-python-headless==4.8.1.78
psycopg2-binary==2.9.9
python-dotenv==1.0.0
gunicorn==21.2.0
Pillow==10.0.0
requests==2.31.0
EOF
fi

# Create necessary directories
mkdir -p static/uploads

# Set execute permissions
chmod +x koyeb-setup.sh

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Build Docker image: docker build -t face-attendance-api ."
echo "2. Test locally: docker run -p 8000:8000 -e PORT=8000 face-attendance-api"
echo "3. Push to Docker Hub: docker tag face-attendance-api YOUR_USERNAME/face-attendance-api"
echo "4. Deploy on Koyeb: Go to Koyeb dashboard → Create App → Deploy from Docker image"
echo ""