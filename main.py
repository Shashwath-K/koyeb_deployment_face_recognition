# main.py - Updated with Chrome DevTools support and debugging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import cv2
import numpy as np
import base64
import os
import uuid
from datetime import datetime
import traceback
import json
import torch

# Import custom modules
from face_engine.detector import FaceDetector
from face_engine.encoder import FaceEncoder
from face_engine.matcher import FaceMatcher
from face_engine.validator import FaceValidator

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'face-recognition-secret-key-2024'

# Initialize components with MobileFaceNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Initialize detector (MTCNN)
detector = FaceDetector(device=device)

# Initialize encoder (MobileFaceNet)
encoder = FaceEncoder(device=device)

# Initialize matcher and validator
matcher = FaceMatcher(threshold=0.6)
validator = FaceValidator()

# Create necessary directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('embeddings', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('models', exist_ok=True)

def get_request_data():
    """Get data from request, handling both JSON and form data"""
    if request.is_json:
        return request.get_json()
    else:
        if request.data:
            try:
                return json.loads(request.data)
            except:
                pass
        return request.form.to_dict()

# ========== CHROME DEVTOOLS SUPPORT ==========

@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def chrome_devtools():
    """Support for Chrome DevTools"""
    return jsonify({
        "name": "Face Recognition System",
        "description": "Face recognition and authentication system",
        "version": "1.0.0",
        "debug": True,
        "endpoints": {
            "capture": "/api/capture",
            "register": "/api/register", 
            "login": "/api/login",
            "stats": "/api/stats",
            "test_camera": "/api/test-camera",
            "debug": "/api/debug"
        }
    })

# ========== DEBUG ROUTES ==========

@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check system status"""
    return jsonify({
        'session': {
            'user_id': session.get('user_id'),
            'captured_faces': len(session.get('captured_faces', [])),
            'session_keys': list(session.keys())
        },
        'system': {
            'device': device,
            'matcher_users': matcher.get_user_count() if hasattr(matcher, 'get_user_count') else 0,
            'matcher_embeddings': matcher.get_total_embeddings() if hasattr(matcher, 'get_total_embeddings') else 0
        },
        'paths': {
            'templates': os.path.exists('templates'),
            'static': os.path.exists('static'),
            'uploads': os.path.exists('static/uploads'),
            'embeddings': os.path.exists('embeddings')
        }
    })

@app.route('/api/debug/reset-session')
def reset_session():
    """Reset session for debugging"""
    session.clear()
    return jsonify({'success': True, 'message': 'Session cleared'})

@app.route('/api/debug/session-info')
def session_info():
    """Get session information"""
    return jsonify({
        'user_id': session.get('user_id'),
        'captured_faces': session.get('captured_faces', []),
        'captured_count': len(session.get('captured_faces', [])),
        'session_id': session.sid if hasattr(session, 'sid') else 'unknown'
    })

# ========== SERVICE WORKER ==========

@app.route('/service-worker.js')
def service_worker():
    """Service worker for PWA support"""
    return app.send_static_file('service-worker.js')

# ========== BASIC ROUTES ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ========== DEBUG PAGES ==========

@app.route('/debug')
def debug_page():
    """Debug page for testing"""
    return render_template('debug.html')

@app.route('/test')
def test_page():
    """Simple test page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            button { padding: 10px 20px; margin: 5px; }
            .result { margin: 20px 0; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>Debug Test Page</h1>
        <button onclick="testAPI()">Test /api/capture</button>
        <button onclick="testRegister()">Test /api/register</button>
        <button onclick="testSession()">Test Session</button>
        <div id="result" class="result"></div>
        
        <script>
        async function testAPI() {
            const result = document.getElementById('result');
            result.innerHTML = 'Testing...';
            
            try {
                // Create a test image
                const canvas = document.createElement('canvas');
                canvas.width = 100;
                canvas.height = 100;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#ff0000';
                ctx.fillRect(0, 0, 100, 100);
                
                const imageData = canvas.toDataURL('image/jpeg');
                
                const response = await fetch('/api/capture', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: imageData})
                });
                
                const data = await response.json();
                result.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            } catch (error) {
                result.innerHTML = 'Error: ' + error.message;
            }
        }
        
        async function testRegister() {
            const result = document.getElementById('result');
            result.innerHTML = 'Testing registration...';
            
            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        name: 'Test User',
                        email: 'test@example.com'
                    })
                });
                
                const data = await response.json();
                result.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            } catch (error) {
                result.innerHTML = 'Error: ' + error.message;
            }
        }
        
        async function testSession() {
            const result = document.getElementById('result');
            result.innerHTML = 'Testing session...';
            
            try {
                const response = await fetch('/api/debug/session-info');
                const data = await response.json();
                result.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            } catch (error) {
                result.innerHTML = 'Error: ' + error.message;
            }
        }
        </script>
    </body>
    </html>
    '''

# ========== API ROUTES ==========

@app.route('/api/capture', methods=['POST'])
def capture_face():
    """Capture face for registration"""
    try:
        data = get_request_data()
        image_data = data.get('image')
        
        print(f"Capture request received. Image data: {'present' if image_data else 'missing'}")
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data'})
        
        # Decode image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image'})
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(image_rgb)
        print(f"Faces detected: {len(faces)}")
        
        if not faces:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        if len(faces) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected'})
        
        # Get face image
        face_pil = faces[0]
        
        # Validate face quality
        face_np = np.array(face_pil)
        is_acceptable, message = validator.is_face_acceptable(face_np)
        if not is_acceptable:
            return jsonify({'success': False, 'message': message})
        
        # Initialize session
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())[:8]
            print(f"New session created with user_id: {session['user_id']}")
        
        if 'captured_faces' not in session:
            session['captured_faces'] = []
        
        # Save face temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{session['user_id']}_{timestamp}.jpg"
        filepath = os.path.join('static/uploads', filename)
        face_pil.save(filepath)
        
        # Store in session
        session['captured_faces'].append(filepath)
        session.modified = True
        
        count = len(session['captured_faces'])
        print(f"Face captured. Total: {count}/3. Session: {session['user_id']}")
        
        return jsonify({
            'success': True,
            'message': f'Face captured ({count}/3)',
            'count': count,
            'user_id': session['user_id']
        })
        
    except Exception as e:
        print(f"Capture error: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/register', methods=['POST'])
def register_user():
    """Complete registration"""
    try:
        data = get_request_data()
        user_id = session.get('user_id')
        captured_faces = session.get('captured_faces', [])
        
        print(f"Register request. User ID: {user_id}, Captured faces: {len(captured_faces)}")
        print(f"Form data: {data}")
        
        # Check if we have a valid user_id
        if not user_id:
            return jsonify({'success': False, 'message': 'No registration session found. Please capture faces first.'})
        
        if len(captured_faces) < 3:
            return jsonify({'success': False, 'message': f'Need 3 face images (currently have {len(captured_faces)})'})
        
        # Get user info
        name = data.get('name', 'Unknown')
        email = data.get('email', '')
        
        if not name or name.strip() == '':
            return jsonify({'success': False, 'message': 'Name is required'})
        
        print(f"Registering user: {name}, {email}")
        
        # Generate embeddings
    
        embeddings=[]
        from PIL import Image
        
        for face_path in captured_faces[:3]:
            try:
                print(f"Processing face: {face_path}")
                face_pil = Image.open(face_path)
                embedding = encoder.encode(face_pil)
                if isinstance(embedding,np.ndarray):
                    embeddings.append(embedding)
                else:
                    embeddings.append(np.array(embedding,dtype=np.float32))
            except Exception as e:
                print(f"Error generating embedding: {e}")
                continue
        
        if not embeddings:
            return jsonify({'success': False, 'message': 'Failed to generate embeddings'})
        embeddings_array=np.array(embeddings,dtype=np.float32)
        # Ensure embeddings is a list of numpy arrays
        
        # Register user in matcher
        user_info = {
            'name': name,
            'email': email,
            'registration_date': datetime.now().isoformat()
        }
        
        matcher.register_user(str(user_id), embeddings_array, user_info)
        
        # Save to disk
        if hasattr(matcher, 'save_database'):
            matcher.save_database()
        
        # Clean up
        for face_path in captured_faces:
            try:
                os.remove(face_path)
            except:
                pass
        
        session.pop('captured_faces', None)
        session.pop('user_id', None)
        
        print(f"Registration successful for user {user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Registration successful!',
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"Registration error: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/login', methods=['POST'])
def login_user():
    """Login with face"""
    try:
        data = get_request_data()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({
                'success': False,
                'authenticated': False,
                'message': 'No image data'
            })
        
        # Decode image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face
        faces = detector.detect_faces(image_rgb)
        
        if not faces:
            return jsonify({
                'success': False,
                'authenticated': False,
                'message': 'No face detected'
            })
        
        # Generate embedding
        face_pil = faces[0]
        probe_embedding = encoder.encode(face_pil)
        
        # Identify user
        matches = matcher.identify_face(probe_embedding, top_k=1)
        
        if matches:
            user_id, similarity = matches[0]
            user_info = matcher.user_info.get(user_id, {})
            
            return jsonify({
                'success': True,
                'authenticated': True,
                'user_id': user_id,
                'name': user_info.get('name', 'User'),
                'similarity': float(similarity),
                'message': f'Welcome {user_info.get("name", "User")}!'
            })
        else:
            return jsonify({
                'success': True,
                'authenticated': False,
                'message': 'User not recognized'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'authenticated': False,
            'message': str(e)
        })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        total_users = matcher.get_user_count() if hasattr(matcher, 'get_user_count') else 0
        total_embeddings = matcher.get_total_embeddings() if hasattr(matcher, 'get_total_embeddings') else 0
        
        return jsonify({
            'total_users': total_users,
            'total_embeddings': total_embeddings,
            'threshold': matcher.threshold if hasattr(matcher, 'threshold') else 0.6,
            'status': 'running',
            'session_active': bool(session.get('user_id')),
            'captured_faces': len(session.get('captured_faces', []))
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/test-camera', methods=['POST'])
def test_camera():
    """Test camera functionality"""
    try:
        data = get_request_data()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data'})
        
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Save test image
            test_path = 'static/uploads/camera_test.jpg'
            cv2.imwrite(test_path, image)
            
            return jsonify({
                'success': True,
                'message': 'Camera working!',
                'image_path': test_path
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to decode image'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'message': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(e):
    print(f"500 Error: {e}")
    traceback.print_exc()
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

# ========== MAIN ==========

if __name__ == '__main__':
    print("=" * 50)
    print("Face Recognition System")
    print("=" * 50)
    
    # Load existing database if available
    if hasattr(matcher, 'load_database'):
        try:
            matcher.load_database()
            print("✓ Loaded existing database")
        except Exception as e:
            print(f"✗ Could not load database: {e}")
            print("✓ Starting with fresh database")
    
    # Create directories
    required_dirs = ['templates', 'static/uploads', 'models', 'embeddings']
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Directory: {dir_path}")
    
    print("\n📊 Debug Endpoints:")
    print(f"  http://127.0.0.1:5001/debug")
    print(f"  http://127.0.0.1:5001/test")
    print(f"  http://127.0.0.1:5001/api/debug")
    print(f"  http://127.0.0.1:5001/api/debug/session-info")
    
    print("\n🎯 Main Pages:")
    print(f"  http://127.0.0.1:5001")
    print(f"  http://127.0.0.1:5001/register")
    print(f"  http://127.0.0.1:5001/login")
    print(f"  http://127.0.0.1:5001/camera")
    print(f"  http://127.0.0.1:5001/dashboard")
    
    print("\n🔧 Chrome DevTools:")
    print(f"  http://127.0.0.1:5001/.well-known/appspecific/com.chrome.devtools.json")
    
    print("\n🚀 Server starting on port 5001...")
    print("=" * 50)
    
    # Run the application
    app.run(debug=True, port=5001, host='0.0.0.0')