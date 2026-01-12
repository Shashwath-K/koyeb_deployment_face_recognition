# Koyeb deployment v2

> Note this version is large in terms of size. Working on implementing a fix along with reducing the complexity with the GPU.

# Face Recognition Attendance System - Koyeb Deployment
<p>To test the capabilities of the PostgreSQL deployed in the Koyeb. Objective is to see if the endpoints in this is working properly across all the platforms without any downtime.</p>

## Deployment

### Method: Deploy from GitHub
1. Fork this repository
2. Go to [Koyeb Dashboard](https://app.koyeb.com/)
3. Click "Create App"
4. Select "GitHub" as source
5. Choose your repository
6. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 120 app:app`
   - **Port**: `8000`
7. Add environment variables from `.env.example`
8. Click "Deploy"

### Method 2: Deploy from Docker Hub
1. Build Docker image:
   ```bash
   docker build -t yourusername/face-attendance .
   docker push yourusername/face-attendance
   ```
## Project Structure
  ``` bash
  *** Incemental Update ***
  project/
  ├── alembic/
  │   ├── versions/
  │   ├── env.py
  │   ├── README
  │   └── script.py.mako
  ├── database/
  │   ├── __init__.py
  │   ├── models.py
  │   ├── connection.py
  │   ├── schema.sql
  │   └── setup.py
  ├── migrations/
  │   └── 001_initial_schema.sql
  ├── scripts/
  │   ├── deploy_schema.py
  │   └── test_schema.py
  ├── requirements.txt
  ├── .env
  ├── .gitignore
  └── README.md
  ```
