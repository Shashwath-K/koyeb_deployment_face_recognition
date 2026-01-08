// static/js/camera.js - UPDATED VERSION WITH SERVER COMMUNICATION
console.log('Loading camera.js...');

class CameraApp {
    constructor() {
        console.log('Creating CameraApp...');
        
        // Initialize properties
        this.cameraStream = null;
        this.currentMode = 'register'; // 'register', 'login', or 'test'
        this.capturedCount = 0;
        this.isCapturing = false;
        
        // Initialize
        this.initialize();
    }
    
    initialize() {
        console.log('Initializing camera app...');
        
        // Get elements
        this.videoElement = document.getElementById('camera-preview');
        this.captureBtn = document.getElementById('capture-btn');
        this.switchBtn = document.getElementById('switch-camera');
        this.testBtn = document.getElementById('test-camera');
        this.statusElement = document.getElementById('status-message');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.resultsDisplay = document.getElementById('results-display');
        
        // Mode buttons
        this.modeButtons = document.querySelectorAll('.mode-btn');
        this.modeRegisterBtn = document.getElementById('mode-register');
        this.modeLoginBtn = document.getElementById('mode-login');
        this.modeTestBtn = document.getElementById('mode-test');
        
        console.log('Elements found:', {
            video: !!this.videoElement,
            captureBtn: !!this.captureBtn,
            status: !!this.statusElement,
            progressFill: !!this.progressFill,
            progressText: !!this.progressText
        });
        
        // Setup event listeners
        if (this.captureBtn) {
            this.captureBtn.addEventListener('click', () => this.captureImage());
        }
        
        if (this.switchBtn) {
            this.switchBtn.addEventListener('click', () => this.switchCamera());
        }
        
        if (this.testBtn) {
            this.testBtn.addEventListener('click', () => this.testCamera());
        }
        
        // Mode selection
        this.modeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setMode(e.target.dataset.mode);
            });
        });
        
        // Start camera
        this.startCamera();
        
        console.log('Camera app initialized successfully!');
    }
    
    async startCamera() {
        try {
            this.cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            });
            this.videoElement.srcObject = this.cameraStream;
            this.updateStatus('Camera ready. Select mode and capture image.', 'info');
        } catch (error) {
            console.error('Camera error:', error);
            this.updateStatus('Camera error: ' + error.message, 'error');
        }
    }
    
    switchCamera() {
        // For simplicity, just reload the page to reset camera
        location.reload();
    }
    
    async captureImage() {
        console.log('Capture button clicked');
        
        if (!this.cameraStream || this.isCapturing) {
            this.updateStatus('Camera not ready or busy', 'error');
            return;
        }
        
        this.isCapturing = true;
        
        try {
            // Create canvas and capture frame
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            // Set canvas dimensions to match video
            canvas.width = this.videoElement.videoWidth;
            canvas.height = this.videoElement.videoHeight;
            
            // Draw video frame to canvas (mirrored)
            context.translate(canvas.width, 0);
            context.scale(-1, 1);
            context.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Handle based on mode
            if (this.currentMode === 'register') {
                await this.handleRegistrationCapture(imageData);
            } else if (this.currentMode === 'login') {
                await this.handleLoginCapture(imageData);
            } else if (this.currentMode === 'test') {
                await this.handleTestCapture(imageData);
            }
            
        } catch (error) {
            console.error('Capture error:', error);
            this.updateStatus('Capture error: ' + error.message, 'error');
        } finally {
            this.isCapturing = false;
        }
    }
    
    async handleRegistrationCapture(imageData) {
        this.updateStatus('Sending face to server...', 'info');
        
        try {
            const response = await fetch('/api/capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            console.log('Capture response:', result);
            
            if (result.success) {
                this.capturedCount = result.count || 0;
                this.updateProgress(this.capturedCount, 3);
                this.updateStatus(result.message, 'success');
                
                // Show registration form after 3 captures
                if (this.capturedCount >= 3) {
                    this.showRegistrationForm();
                }
            } else {
                this.updateStatus(result.message, 'error');
            }
        } catch (error) {
            console.error('Error capturing face:', error);
            this.updateStatus('Network error: ' + error.message, 'error');
        }
    }
    
    async handleLoginCapture(imageData) {
        this.updateStatus('Authenticating...', 'info');
        
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            console.log('Login response:', result);
            
            if (result.success && result.authenticated) {
                this.showResults(`
                    <h3>✅ Login Successful!</h3>
                    <p>Welcome back, ${result.name}!</p>
                    <p>Similarity: ${(result.similarity * 100).toFixed(2)}%</p>
                    <a href="/" class="btn btn-primary">Go to Home</a>
                `, 'success');
                this.updateStatus(`Welcome ${result.name}!`, 'success');
            } else {
                this.showResults(`
                    <h3>❌ Login Failed</h3>
                    <p>${result.message || 'User not recognized'}</p>
                    <p>Please try again or register if you're a new user.</p>
                `, 'error');
                this.updateStatus(result.message || 'Authentication failed', 'error');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.updateStatus('Network error: ' + error.message, 'error');
        }
    }
    
    async handleTestCapture(imageData) {
        this.updateStatus('Testing camera...', 'info');
        
        try {
            const response = await fetch('/api/test-camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            console.log('Test response:', result);
            
            if (result.success) {
                this.showResults(`
                    <h3>✅ Camera Test Successful!</h3>
                    <p>Your camera is working properly.</p>
                    ${result.image_path ? `<p>Test image saved</p>` : ''}
                `, 'success');
                this.updateStatus('Camera test successful!', 'success');
            } else {
                this.showResults(`
                    <h3>❌ Camera Test Failed</h3>
                    <p>${result.message}</p>
                `, 'error');
                this.updateStatus('Camera test failed', 'error');
            }
        } catch (error) {
            console.error('Test error:', error);
            this.updateStatus('Network error: ' + error.message, 'error');
        }
    }
    
    async testCamera() {
        // Quick test without capturing
        this.updateStatus('Testing camera connection...', 'info');
        
        try {
            // Try to access camera again
            await navigator.mediaDevices.getUserMedia({ video: true });
            this.updateStatus('✅ Camera is working!', 'success');
        } catch (error) {
            this.updateStatus('❌ Camera test failed: ' + error.message, 'error');
        }
    }
    
    setMode(mode) {
        this.currentMode = mode;
        this.capturedCount = 0;
        
        // Update UI
        this.modeButtons.forEach(btn => {
            if (btn.dataset.mode === mode) {
                btn.classList.add('active');
                btn.classList.remove('btn-secondary');
                btn.classList.add('btn-primary');
            } else {
                btn.classList.remove('active');
                btn.classList.remove('btn-primary');
                btn.classList.add('btn-secondary');
            }
        });
        
        // Reset for new mode
        if (mode === 'register') {
            this.updateProgress(0, 3);
            this.updateStatus('Ready for registration. Capture 3 face images.', 'info');
        } else if (mode === 'login') {
            this.updateStatus('Ready for login. Look at the camera and capture your face.', 'info');
        } else if (mode === 'test') {
            this.updateStatus('Ready for camera test. Capture an image to test.', 'info');
        }
        
        // Hide previous results
        if (this.resultsDisplay) {
            this.resultsDisplay.style.display = 'none';
        }
    }
    
    updateProgress(current, total) {
        const percentage = (current / total) * 100;
        
        if (this.progressFill) {
            this.progressFill.style.width = percentage + '%';
        }
        
        if (this.progressText) {
            this.progressText.textContent = `${current}/${total}`;
        }
        
        console.log(`Progress: ${current}/${total} (${percentage}%)`);
    }
    
    updateStatus(message, type = 'info') {
        if (this.statusElement) {
            this.statusElement.textContent = message;
            
            // Update styling based on type
            this.statusElement.style.borderLeftColor = '#4299e1'; // default blue
            this.statusElement.style.background = '#bee3f8'; // default light blue
            
            if (type === 'error') {
                this.statusElement.style.borderLeftColor = '#e53e3e';
                this.statusElement.style.background = '#fed7d7';
            } else if (type === 'success') {
                this.statusElement.style.borderLeftColor = '#38a169';
                this.statusElement.style.background = '#c6f6d5';
            } else if (type === 'warning') {
                this.statusElement.style.borderLeftColor = '#d69e2e';
                this.statusElement.style.background = '#feebc8';
            }
        }
        
        console.log('Status:', message, 'Type:', type);
    }
    
    showRegistrationForm() {
        // Create registration form if it doesn't exist
        if (!document.getElementById('registration-form-container')) {
            const formHTML = `
                <div id="registration-form-container" style="margin-top: 30px; padding: 20px; background: #f7fafc; border-radius: 8px;">
                    <h3>Complete Registration</h3>
                    <p>3 faces captured! Enter your details to complete registration.</p>
                    <form id="registration-form">
                        <div class="form-group" style="margin-bottom: 15px;">
                            <label for="reg-name" style="display: block; margin-bottom: 5px; font-weight: 500;">Full Name:</label>
                            <input type="text" id="reg-name" name="name" required 
                                   style="width: 100%; padding: 10px; border: 1px solid #e2e8f0; border-radius: 4px;">
                        </div>
                        <div class="form-group" style="margin-bottom: 15px;">
                            <label for="reg-email" style="display: block; margin-bottom: 5px; font-weight: 500;">Email (optional):</label>
                            <input type="email" id="reg-email" name="email" 
                                   style="width: 100%; padding: 10px; border: 1px solid #e2e8f0; border-radius: 4px;">
                        </div>
                        <button type="submit" class="btn btn-primary" style="width: 100%; padding: 12px;">
                            Complete Registration
                        </button>
                    </form>
                </div>
            `;
            
            // Insert after progress bar
            const statusElement = this.statusElement.parentNode;
            statusElement.insertAdjacentHTML('afterend', formHTML);
            
            // Add form submit handler
            document.getElementById('registration-form').addEventListener('submit', (e) => this.handleRegistrationSubmit(e));
        }
        
        this.updateStatus('✅ 3 faces captured! Enter your details below.', 'success');
    }
    
    async handleRegistrationSubmit(e) {
        e.preventDefault();
        
        const name = document.getElementById('reg-name').value.trim();
        const email = document.getElementById('reg-email').value.trim();
        
        if (!name) {
            this.updateStatus('Please enter your name', 'error');
            return;
        }
        
        try {
            this.updateStatus('Registering...', 'info');
            
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    name: name,
                    email: email 
                })
            });
            
            const result = await response.json();
            console.log('Registration response:', result);
            
            if (result.success) {
                this.showResults(`
                    <h3>✅ Registration Successful!</h3>
                    <p>Welcome ${name}!</p>
                    <p>Your user ID: ${result.user_id}</p>
                    <p>${result.message}</p>
                    <a href="/" class="btn btn-primary">Go to Home</a>
                `, 'success');
                
                // Hide form
                const formContainer = document.getElementById('registration-form-container');
                if (formContainer) {
                    formContainer.style.display = 'none';
                }
                
                // Reset for next user
                this.capturedCount = 0;
                this.updateProgress(0, 3);
                this.updateStatus('Registration complete! You can register another user.', 'success');
            } else {
                this.showResults(`
                    <h3>❌ Registration Failed</h3>
                    <p>${result.message}</p>
                    <p>Please try capturing faces again.</p>
                `, 'error');
                this.updateStatus('Registration failed: ' + result.message, 'error');
            }
        } catch (error) {
            console.error('Registration error:', error);
            this.updateStatus('Registration error: ' + error.message, 'error');
        }
    }
    
    showResults(content, type = 'info') {
        if (this.resultsDisplay) {
            this.resultsDisplay.innerHTML = content;
            this.resultsDisplay.className = '';
            this.resultsDisplay.classList.add(type === 'success' ? 'success-result' : 
                                            type === 'error' ? 'error-result' : 'info-result');
            this.resultsDisplay.style.display = 'block';
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, creating camera app...');
    try {
        window.cameraApp = new CameraApp();
        console.log('Camera app created successfully:', window.cameraApp);
        
        // Expose for debugging
        window.testCamera = () => window.cameraApp.testCamera();
        window.captureTest = () => window.cameraApp.captureImage();
        
    } catch (error) {
        console.error('Failed to create camera app:', error);
    }
});