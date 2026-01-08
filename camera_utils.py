# camera_utils.py - Camera utilities
import cv2
import numpy as np
import base64
class CameraUtils:
    @staticmethod
    def get_camera_list():
        """Get list of available cameras"""
        cameras = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cameras.append({
                        'index': i,
                        'name': f'Camera {i}',
                        'resolution': f'{frame.shape[1]}x{frame.shape[0]}'
                    })
                cap.release()
        return cameras
    
    @staticmethod
    def capture_frame(camera_index=0, resolution=(640, 480)):
        """Capture frame from camera"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        return None
    
    @staticmethod
    def frame_to_base64(frame, quality=80):
        """Convert frame to base64"""
        if frame is None:
            return None
        
        try:
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
            
            if not success or encoded_image is None:
                return None
            
            # Convert numpy array to bytes
            image_bytes = encoded_image.tobytes()
            
            # Convert to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        
        except Exception as e:
            print(f"Error converting frame to base64: {e}")
            return None