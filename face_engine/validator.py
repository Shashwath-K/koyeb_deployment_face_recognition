# validator.py - Input validation and quality assessment
import numpy as np
import cv2
from typing import Tuple, Dict, List

class FaceValidator:
    def __init__(self):
        """Initialize face validator"""
        # Quality thresholds
        self.min_face_size = (100, 100)  # Minimum face size in pixels
        self.max_face_size = (500, 500)  # Maximum face size in pixels
        self.min_brightness = 30         # Minimum brightness value
        self.max_brightness = 220        # Maximum brightness value
        self.min_sharpness = 10          # Minimum sharpness score
        self.min_contrast = 20           # Minimum contrast
        
    def validate_face_size(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate face size.
        
        Args:
            face_image: Face image
            
        Returns:
            Tuple of (is_valid, message)
        """
        height, width = face_image.shape[:2]
        
        if height < self.min_face_size[0] or width < self.min_face_size[1]:
            return False, f"Face too small: {width}x{height}. Minimum: {self.min_face_size[1]}x{self.min_face_size[0]}"
        
        if height > self.max_face_size[0] or width > self.max_face_size[1]:
            return False, f"Face too large: {width}x{height}. Maximum: {self.max_face_size[1]}x{self.max_face_size[0]}"
        
        return True, f"Face size OK: {width}x{height}"
    
    def validate_brightness(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image brightness.
        
        Args:
            face_image: Face image
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Convert to grayscale for brightness calculation
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        mean_brightness = np.mean(np.array(gray, dtype=np.float32))
        
        if mean_brightness < self.min_brightness:
            return False, f"Image too dark: {mean_brightness:.1f}. Minimum: {self.min_brightness}"
        
        if mean_brightness > self.max_brightness:
            return False, f"Image too bright: {mean_brightness:.1f}. Maximum: {self.max_brightness}"
        
        return True, f"Brightness OK: {mean_brightness:.1f}"
    
    def validate_sharpness(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image sharpness using Laplacian variance.
        
        Args:
            face_image: Face image
            
        Returns:
            Tuple of (is_valid, message)
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < self.min_sharpness:
            return False, f"Image too blurry: {laplacian_var:.1f}. Minimum: {self.min_sharpness}"
        
        return True, f"Sharpness OK: {laplacian_var:.1f}"
    
    def validate_contrast(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image contrast.
        
        Args:
            face_image: Face image
            
        Returns:
            Tuple of (is_valid, message)
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Calculate contrast as standard deviation of pixel intensities
        contrast = np.std(gray)
        
        if contrast < self.min_contrast:
            return False, f"Contrast too low: {contrast:.1f}. Minimum: {self.min_contrast}"
        
        return True, f"Contrast OK: {contrast:.1f}"
    
    def validate_face_angle(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Simple face angle validation using face symmetry.
        
        Args:
            face_image: Face image
            
        Returns:
            Tuple of (is_valid, message)
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Split face into left and right halves
        height, width = gray.shape
        mid = width // 2
        
        left_half = gray[:, :mid]
        right_half = gray[:, mid:]
        
        # Flip right half for comparison
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize if dimensions don't match
        if left_half.shape != right_half_flipped.shape:
            right_half_flipped = cv2.resize(right_half_flipped, 
                                           (left_half.shape[1], left_half.shape[0]))
        
        # Calculate similarity between halves
        mse = np.mean((left_half - right_half_flipped) ** 2)
        
        # MSE threshold for frontal face
        mse_threshold = 1000
        
        if mse > mse_threshold:
            return False, f"Face not frontal enough. Symmetry score: {mse:.1f}"
        
        return True, f"Face angle OK. Symmetry score: {mse:.1f}"
    
    def validate_face_quality(self, face_image: np.ndarray) -> Dict[str, Tuple[bool, str]]:
        """
        Perform comprehensive face quality validation.
        
        Args:
            face_image: Face image
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Perform all validations
        results['size'] = self.validate_face_size(face_image)
        results['brightness'] = self.validate_brightness(face_image)
        results['sharpness'] = self.validate_sharpness(face_image)
        results['contrast'] = self.validate_contrast(face_image)
        results['angle'] = self.validate_face_angle(face_image)
        
        # Overall validation
        all_valid = all(result[0] for result in results.values())
        results['overall'] = (all_valid, "All checks passed" if all_valid else "Some checks failed")
        
        return results
    
    def is_face_acceptable(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Quick check if face is acceptable for registration.
        
        Args:
            face_image: Face image
            
        Returns:
            Tuple of (is_acceptable, message)
        """
        validations = self.validate_face_quality(face_image)
        
        # Check all validations except angle for quick acceptance
        critical_validations = ['size', 'brightness', 'sharpness']
        
        for validation in critical_validations:
            is_valid, message = validations[validation]
            if not is_valid:
                return False, f"{validation}: {message}"
        
        return True, "Face is acceptable for registration"
    
    def get_quality_score(self, face_image: np.ndarray) -> float:
        """
        Calculate overall quality score (0-100).
        
        Args:
            face_image: Face image
            
        Returns:
            Quality score
        """
        validations = self.validate_face_quality(face_image)
        
        # Count passed validations
        passed = sum(1 for result in validations.values() if result[0])
        total = len(validations) - 1  # Exclude overall
        
        # Calculate percentage
        quality_score = (passed / total) * 100 if total > 0 else 0
        
        return quality_score
    
    def preprocess_for_better_quality(self, face_image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to improve face image quality.
        
        Args:
            face_image: Original face image
            
        Returns:
            Preprocessed face image
        """
        processed = face_image.copy()
        
        # Convert to grayscale if needed for processing
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
        
        # Apply histogram equalization for better contrast
        gray_eq = cv2.equalizeHist(gray)
        
        # Apply mild sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        gray_sharp = cv2.filter2D(gray_eq, -1, kernel)
        
        # Convert back to BGR if original was color
        if len(face_image.shape) == 3:
            processed = cv2.cvtColor(gray_sharp, cv2.COLOR_GRAY2BGR)
        else:
            processed = gray_sharp
        
        return processed