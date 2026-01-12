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
> Deployment for the current version faces the following problems, The logs specify the exact error
``` bash
[17:42:31] Koyeb Runtime Logs - Application: face-attendance-system
[17:42:31] ============================================================

[17:42:31] [INFO] Starting application build process...
[17:42:33] [INFO] Installing Python dependencies...
[17:42:45] [SUCCESS] Requirements installed: 45 packages

[17:42:47] [WARNING] Build size exceeds 1.8GB (limit: 2GB)
[17:42:47] [INFO] Build size breakdown:
[17:42:47]          tensorflow-2.13.0: 780MB
[17:42:47]          dlib-19.24.2: 320MB
[17:42:47]          face-recognition-1.3.0: 180MB
[17:42:47]          opencv-python-4.8.1: 120MB
[17:42:47]          Other dependencies: 450MB
[17:42:47] [TOTAL] 1.85GB / 2.00GB (92.5%)

[17:42:49] [INFO] Starting application container...
[17:42:50] [INFO] Container memory limit: 256MB
[17:42:51] [INFO] Container CPU limit: 0.5 vCPU

[17:42:53] [ERROR] Application failed to start
[17:42:53] [ERROR] Exit code: 137 (Out of Memory)
[17:42:53] [CRITICAL] Memory usage at crash:
[17:42:53]          Python runtime: 120MB
[17:42:53]          TensorFlow loading: +180MB
[17:42:53]          Face recognition model: +220MB
[17:42:53]          Dlib shape predictor: +85MB
[17:42:53] [ESTIMATED] 605MB required > 256MB limit

[17:42:55] [INFO] Attempting automatic restart...
[17:42:58] [ERROR] Restart failed - Same error (OOM)

[17:43:02] [INFO] Koyeb scheduler triggered scale-to-zero
[17:43:02] [WARNING] Application suspended due to resource constraints

[17:43:05] [DIAGNOSTIC] Resource constraint analysis:
[17:43:05] --------------------------------------------------
[17:43:05] CPU:    0.5 vCPU sufficient for Python/Flask
[17:43:05] STORAGE: 1.85GB/2.00GB (near limit)
[17:43:05] MEMORY:  605MB required / 256MB available (236% over)
[17:43:05] 
[17:43:05] PRIMARY BOTTLENECKS:
[17:43:05] 1. Deep learning models exceed RAM capacity
[17:43:05] 2. Model files consume excessive storage
[17:43:05] 3. Multiple heavy ML libraries conflict

[17:43:07] [RECOMMENDATION] Required changes:
[17:43:07] --------------------------------------------------
[17:43:07] 1. Replace TensorFlow with lighter alternative
[17:43:07] 2. Use OpenCV DNN for face detection (no dlib)
[17:43:07] 3. Implement model quantization
[17:43:07] 4. Use external model storage (AWS S3, etc.)
[17:43:07] 5. Implement lazy loading of models

[17:43:10] [STATUS] Application: CRASH_LOOP_BACKOFF
[17:43:10] [ACTION] Manual intervention required
[17:43:10] ============================================================

[17:43:15] Koyeb Infrastructure Logs:
[17:43:15] --------------------------------------------------
[17:43:15] Instance Type: Free Tier (Starter)
[17:43:15] Allocated Resources:
[17:43:15]   - vCPU: 0.5 (shared)
[17:43:15]   - RAM: 256MB (non-swappable)
[17:43:15]   - Storage: 2GB (ephemeral)
[17:43:15]   - Network: 100MB bandwidth
[17:43:15] 
[17:43:15] Limitations:
[17:43:15]   - No swap space available
[17:43:15]   - CPU throttled after 100% usage
[17:43:15]   - Cold starts after inactivity
[17:43:15]   - Build timeout: 10 minutes

[17:43:20] [NOTE] For face recognition on free tier:
[17:43:20] Consider:
[17:43:20] 1. Upgrading to Hobby ($9/mo - 1GB RAM)
[17:43:20] 2. Using serverless functions for ML
[17:43:20] 3. External face recognition API
[17:43:20] 4. Client-side face encoding
```

- To fix this issue the next approach is to use nGrok as suggested by the re-search to test out the mobile apllications and as per the suggestion from the guide the best and only way for the app to communicate accross all the platforms is to use the FAST API instead of FLASK API which has some restrictions on certain platforms.

## Deployment Failure Analysis and Solution Strategy

### Identified Issues with Current Deployment

Resource Constraints on Koyeb Free Tier:
- Memory Limit Exceeded: Application requires ~605MB RAM, but free tier provides only 256MB
- Storage Near Capacity: Build size at 1.85GB out of 2GB limit (92.5% utilization)
- Heavy Dependencies: TensorFlow (780MB), dlib (320MB), and face-recognition (180MB) libraries are too large

Primary Failure Points:
- OOM (Out of Memory) Errors: Python process killed with exit code 137
- Multiple Heavy ML Libraries: Simultaneous loading of TensorFlow, dlib, and OpenCV exceeds memory limits
- Insufficient CPU Allocation: 0.5 vCPU throttles under ML workload
- No Swap Space: Memory cannot be paged to disk on free tier

### Solution Implementation Plan

Immediate Technical Adjustments:
- Replace Flask with FastAPI for better cross-platform compatibility and performance
- Implement ngrok tunneling for mobile application testing across networks
- Use lightweight alternatives to heavy ML libraries
- Optimize Docker image with slim base image and minimal dependencies
- Implement lazy loading of face recognition models

Architectural Changes:
- Separate face detection and recognition into microservices
- Consider serverless functions for compute-intensive operations
- Implement client-side face encoding to reduce server load
- Use external storage (S3, Cloud Storage) for model files

Resource Optimization:
- Quantize face recognition models for smaller memory footprint
- Implement request queuing to prevent concurrent memory spikes
- Add health checks with graceful degradation
- Monitor memory usage with automatic scaling triggers

Testing Strategy with ngrok:
- Use ngrok to create secure tunnels for mobile app testing
- Enable WebSocket support for real-time face detection
- Configure CORS properly for cross-origin requests
- Implement authentication for tunneled endpoints

Deployment Workflow:
1. Local development with optimized dependencies
2. ngrok tunnel testing with mobile applications
3. Staging deployment with resource monitoring
4. Production deployment with auto-scaling configuration

Expected Outcomes:
- Reduce memory usage to under 200MB
- Decrease build size to under 1GB
- Maintain sub-second response times
- Enable reliable cross-platform communication
- Eliminate OOM crashes on free tier

### Migration Checklist
- [ ] Replace Flask endpoints with FastAPI
- [ ] Configure ngrok for local testing
- [ ] Optimize Dockerfile with multi-stage builds
- [ ] Implement lightweight face detection (OpenCV DNN)
- [ ] Add memory monitoring and alerting
- [ ] Test with mobile applications via ngrok tunnel
- [ ] Document API changes for mobile team

