import cv2
import numpy as np
import hashlib
import easyocr
from typing import Tuple, Optional, Dict

class OCRProcessor:
    """Handles OCR operations with rotation support and confidence tracking."""
    
    def __init__(self, languages: list = None):
        """Initialize OCR processor with specified languages."""
        self.reader = easyocr.Reader(languages or ['en'])
        self._cache = {}  # Cache for repeated OCR operations
    
    def extract_text_with_rotations(self, crop_img: np.ndarray, 
                                   obb_points: np.ndarray,
                                   early_exit_confidence: float = 0.9) -> Tuple[Optional[str], float]:
        """
        Perform OCR on image with multiple rotation attempts for better accuracy.
        Optimized with early exit and caching.
        
        Args:
            crop_img: Cropped image for OCR
            obb_points: OBB coordinates for orientation
            early_exit_confidence: Stop trying rotations if this confidence is reached
            
        Returns:
            Tuple of (best_text, best_confidence)
        """
        # Pre-filter: Check if crop is viable for OCR
        if not self._is_viable_for_ocr(crop_img):
            print("  - OCR Result: Crop filtered out (too small/poor quality).")
            return None, 0.0
        
        # Generate cache key for this crop
        crop_hash = self._generate_crop_hash(crop_img)
        if crop_hash in self._cache:
            cached_result = self._cache[crop_hash]
            print(f"  - OCR Result: Using cached result '{cached_result[0]}' (Confidence: {cached_result[1]:.3f})")
            return cached_result
        
        best_text, best_confidence = None, 0.0
        
        # Get optimal rotation strategy based on OBB
        rotations = self.get_optimal_ocr_strategy(obb_points)
        
        # Try original orientation first
        text, confidence = self._perform_ocr(crop_img)
        if confidence > best_confidence:
            best_text, best_confidence = text, confidence
        
        # Early exit if we have high confidence
        if best_confidence >= early_exit_confidence:
            print(f"  - OCR Result: Early exit with high confidence '{best_text}' ({best_confidence:.3f})")
            self._cache[crop_hash] = (best_text, best_confidence)
            return best_text, best_confidence
        
        # Only try rotations if initial result is poor and crop quality is decent
        if best_confidence < 0.3 and self._has_decent_quality(crop_img):
            for angle in rotations:
                try:
                    rotated_crop = self._rotate_image(crop_img, angle)
                    text, confidence = self._perform_ocr(rotated_crop)
                    
                    if confidence > best_confidence:
                        best_text, best_confidence = text, confidence
                        print(f"  - Better OCR found at {angle}° rotation")
                        
                        # Early exit if we reach good confidence
                        if confidence >= early_exit_confidence:
                            print(f"  - Early exit at {angle}° rotation")
                            break
                        
                except Exception as e:
                    print(f"  - Rotation {angle}° failed: {e}")
                    continue
        
        # Cache the result
        if best_text and best_confidence > 0:
            self._cache[crop_hash] = (best_text, best_confidence)
            print(f"  - OCR Result: '{best_text}' (Confidence: {best_confidence:.3f})")
            return best_text, best_confidence
        else:
            print("  - OCR Result: No text found.")
            return None, 0.0
    
    def _is_viable_for_ocr(self, crop_img: np.ndarray) -> bool:
        """Pre-filter crops that are unlikely to yield good OCR results."""
        h, w = crop_img.shape[:2]
        
        # Size checks
        if h < 10 or w < 10:
            return False
        
        if h * w < 400:  # Minimum area
            return False
        
        # Aspect ratio check
        aspect_ratio = max(w/h, h/w)
        if aspect_ratio > 10.0:
            return False
        
        # Basic quality checks
        if not self._has_sufficient_contrast(crop_img):
            return False
        
        return True
    
    def _has_decent_quality(self, crop_img: np.ndarray) -> bool:
        """Check if crop has decent quality for rotation attempts."""
        # Convert to grayscale for analysis
        if len(crop_img.shape) == 3:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_img
        
        # Check for blur (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > 100  # Threshold for non-blurry images
    
    def _has_sufficient_contrast(self, crop_img: np.ndarray) -> bool:
        """Check if crop has sufficient contrast for OCR."""
        if len(crop_img.shape) == 3:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_img
        
        # Calculate standard deviation as contrast measure
        contrast = np.std(gray)
        return contrast > 20.0
    
    def _generate_crop_hash(self, crop_img: np.ndarray) -> str:
        """Generate a hash for the crop image for caching."""
        # Resize to small size for consistent hashing
        small_crop = cv2.resize(crop_img, (32, 32))
        return hashlib.md5(small_crop.tobytes()).hexdigest()
    
    def _perform_ocr(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Perform OCR on a single image."""
        try:
            results = self.reader.readtext(image)
            if not results:
                return None, 0.0
            
            # Find result with highest confidence
            best_result = max(results, key=lambda x: x[2])
            return best_result[1].strip(), best_result[2]
            
        except Exception as e:
            print(f"  - OCR Error: {e}")
            return None, 0.0
    
    @staticmethod
    def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle."""
        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        return cv2.rotate(image, rotation_map[angle])

    def get_optimal_ocr_strategy(self, obb_points: np.ndarray) -> Tuple[int, ...]:
        """Determine optimal OCR strategy based on OBB orientation."""
        orientation = OCRProcessor.calculate_obb_orientation(obb_points)
        
        if orientation == "vertical_normal":
            return (0,)
        elif orientation == "vertical_inverted":
            return (180,)
        elif orientation == "horizontal":
            return (90, 270)
        else:
            return (0, 90, 180, 270)

    @staticmethod
    def calculate_obb_orientation(obb_points: np.ndarray) -> str:
        """
        Calculate the orientation of the OBB based on its dimensions.
        Assumes OBB points are ordered (top-left, top-right, bottom-right, bottom-left)
        or similar, such that adjacent points define width/height.
        """
        points = obb_points.reshape(4, 2)
        
        # Calculate side lengths
        side1_len = np.linalg.norm(points[0] - points[1]) # Top side
        side2_len = np.linalg.norm(points[1] - points[2]) # Right side

        # Determine if it's more vertical or horizontal
        if side2_len > side1_len:
            # Potentially vertical. Check if the text is upright or inverted.
            # This is a simplification; a more robust check would involve
            # analyzing the angle of the longest side relative to the y-axis.
            # For book spines, we assume vertical_normal for now.
            return "vertical_normal"
        else:
            return "horizontal"
