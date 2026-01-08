import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
from PIL import Image

class FaceDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize MTCNN face detector"""
        self.device = device
        self.detector = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
            margin=20
        )
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: numpy array or PIL Image
        
        Returns:
            List of detected face images (PIL Images)
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # RGB
                pass
            image = Image.fromarray(image)
        
        # Detect faces - MTCNN returns (boxes, probs, landmarks)
        # We only need boxes and probs, so we can use underscore for landmarks
        boxes, probs, _ = self.detector.detect(image, landmarks=True)
        
        faces = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Check probability threshold
                if probs[i] > 0.9:  # You can adjust this threshold
                    # Extract face
                    face = image.crop(box)
                    # Resize to 112x112 (MobileFaceNet input size)
                    face = face.resize((112, 112), Image.Resampling.BILINEAR)
                    faces.append(face)
        
        return faces
    
    def detect_largest_face(self, image):
        """
        Detect and return the largest face in the image
        
        Args:
            image: numpy array or PIL Image
        
        Returns:
            Single face image (PIL Image) or None
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Return the first face (MTCNN returns faces in order of detection confidence)
        return faces[0]