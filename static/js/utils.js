// utils.js - Utility functions

// Show notification
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    
    if (type === 'success') {
        notification.style.background = '#10b981';
    } else if (type === 'error') {
        notification.style.background = '#ef4444';
    } else if (type === 'warning') {
        notification.style.background = '#f59e0b';
    } else {
        notification.style.background = '#4f46e5';
    }
    
    document.body.appendChild(notification);
    
    // Remove after duration
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Format date
function formatDate(date) {
    return new Date(date).toLocaleString();
}

// Validate email
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// Get camera stream
async function getCameraStream(deviceId = null, width = 640, height = 480) {
    try {
        const constraints = {
            video: {
                width: { ideal: width },
                height: { ideal: height },
                facingMode: 'user'
            },
            audio: false
        };
        
        if (deviceId) {
            constraints.video.deviceId = { exact: deviceId };
        }
        
        return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
        console.error('Error accessing camera:', error);
        throw error;
    }
}

// Stop camera stream
function stopCameraStream(stream) {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
}

// Capture frame from video
function captureFrame(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    
    // Mirror for selfie view
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Export functions
window.Utils = {
    showNotification,
    formatDate,
    validateEmail,
    getCameraStream,
    stopCameraStream,
    captureFrame
};