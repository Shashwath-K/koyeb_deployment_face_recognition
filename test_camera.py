# camera_test.py - Test camera functionality
import cv2
import sys

def test_camera():
    # Try different camera indices
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
        
        if cap.isOpened():
            print(f"✓ Camera found at index {i}")
            ret, frame = cap.read()
            
            if ret:
                print(f"  Frame size: {frame.shape}")
                cv2.imshow(f'Camera {i}', frame)
                cv2.waitKey(1000)  # Show for 1 second
                cv2.destroyAllWindows()
            else:
                print(f"  Could not read frame from camera {i}")
            
            cap.release()
        else:
            print(f"✗ No camera at index {i}")

if __name__ == '__main__':
    print("Testing camera...")
    test_camera()