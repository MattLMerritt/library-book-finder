import cv2
import numpy as np
from typing import Tuple

class CropExtractor:
    """Handles extraction of crops from frames for both OBB and regular boxes."""
    
    @staticmethod
    def extract_obb_crop(frame: np.ndarray, obb_points: np.ndarray) -> np.ndarray:
        """Extract rotated crop from frame using OBB coordinates."""
        try:
            points = obb_points.reshape(4, 2).astype(np.float32)
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            
            width, height = int(rect[1][0]), int(rect[1][1])
            
            if width < 10 or height < 10:
                raise ValueError("Rotated crop too small")
            
            # Define destination points for perspective transform
            dst_pts = np.array([
                [0, height-1], [0, 0], [width-1, 0], [width-1, height-1]
            ], dtype=np.float32)
            
            # Apply perspective transform
            M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)
            return cv2.warpPerspective(frame, M, (width, height))
            
        except Exception as e:
            print(f"OBB extraction failed: {e}, using fallback")
            return CropExtractor._extract_fallback_crop(frame, obb_points)
    
    @staticmethod
    def extract_regular_crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Extract regular crop from frame using bounding box coordinates."""
        x1, y1, x2, y2 = box
        return frame[max(0, y1):y2, max(0, x1):x2]
    
    @staticmethod
    def _extract_fallback_crop(frame: np.ndarray, obb_points: np.ndarray) -> np.ndarray:
        """Fallback method for OBB crop extraction."""
        try:
            coords = obb_points.flatten()
            x_coords, y_coords = coords[::2], coords[1::2]
            
            x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
            y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))
            
            # Ensure valid bounds
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]
            else:
                return np.zeros((20, 20, 3), dtype=np.uint8)
                
        except Exception:
            return np.zeros((20, 20, 3), dtype=np.uint8)
