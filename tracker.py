"""
Optimized YOLO Object Tracker with OCR
Supports both regular bounding boxes and oriented bounding boxes (OBB).
Features confidence-based OCR retry system for improved text extraction accuracy.
"""

import cv2
import numpy as np
import random
from typing import Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
from ultralytics import YOLO
import easyocr


@dataclass
class TrackerConfig:
    """Configuration class for tracker parameters."""
    video_path: str
    model_path: str
    tracker_config: str
    confidence_threshold: float = 0.7
    frame_resize_factor: float = 0.5
    min_crop_size: int = 10
    ocr_rotations: Tuple[int, ...] = (90, 180, 270)


class OCRProcessor:
    """Handles OCR operations with rotation support and confidence tracking."""
    
    def __init__(self, languages: list = None):
        """Initialize OCR processor with specified languages."""
        self.reader = easyocr.Reader(languages or ['en'])
        self._cache = {}  # Simple cache for repeated OCR operations
    
    def extract_text_with_rotations(self, crop_img: np.ndarray, 
                                   rotations: Tuple[int, ...] = (90, 180, 270)) -> Tuple[Optional[str], float]:
        """
        Perform OCR on image with multiple rotation attempts for better accuracy.
        
        Args:
            crop_img: Cropped image for OCR
            rotations: Angles to try for rotation
            
        Returns:
            Tuple of (best_text, best_confidence)
        """
        if crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
            print("  - OCR Result: Crop too small, skipping.")
            return None, 0.0
        
        best_text, best_confidence = None, 0.0
        
        # Try original orientation first
        text, confidence = self._perform_ocr(crop_img)
        if confidence > best_confidence:
            best_text, best_confidence = text, confidence
        
        # Try rotated versions
        for angle in rotations:
            try:
                rotated_crop = self._rotate_image(crop_img, angle)
                text, confidence = self._perform_ocr(rotated_crop)
                
                if confidence > best_confidence:
                    best_text, best_confidence = text, confidence
                    print(f"  - Better OCR found at {angle}° rotation")
                    
            except Exception as e:
                print(f"  - Rotation {angle}° failed: {e}")
                continue
        
        if best_text and best_confidence > 0:
            print(f"  - OCR Result: '{best_text}' (Confidence: {best_confidence:.3f})")
            return best_text, best_confidence
        else:
            print("  - OCR Result: No text found.")
            return None, 0.0
    
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


class Visualizer:
    """Handles visualization of detections on frames."""
    
    @staticmethod
    def draw_obb(frame: np.ndarray, obb_points: np.ndarray, track_id: int, 
                 class_name: str, color: Tuple[int, int, int]) -> None:
        """Draw oriented bounding box with label."""
        points = obb_points.reshape(4, 2).astype(int)
        cv2.polylines(frame, [points], True, color, 2)
        
        label = f"ID:{track_id} {class_name}"
        label_pos = (int(points[0][0]), int(points[0][1]) - 10)
        Visualizer._draw_label(frame, label, label_pos, color)
    
    @staticmethod
    def draw_bbox(frame: np.ndarray, box: np.ndarray, track_id: int, 
                  class_name: str, color: Tuple[int, int, int]) -> None:
        """Draw regular bounding box with label."""
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID:{track_id} {class_name}"
        label_pos = (x1, y1 - 5)
        Visualizer._draw_label(frame, label, label_pos, color)
    
    @staticmethod
    def _draw_label(frame: np.ndarray, label: str, pos: Tuple[int, int], 
                   color: Tuple[int, int, int]) -> None:
        """Draw label with background."""
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (pos[0], pos[1] - text_height - 5), 
                     (pos[0] + text_width, pos[1]), 
                     color, -1)
        
        # Draw text
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


class DetectionProcessor:
    """Processes detections and manages OCR operations."""
    
    def __init__(self, ocr_processor: OCRProcessor, config: TrackerConfig):
        self.ocr_processor = ocr_processor
        self.config = config
        self.crop_extractor = CropExtractor()
        self.visualizer = Visualizer()
    
    def process_detection(self, frame: np.ndarray, detection_data: dict, 
                         track_ocr_confidence: Dict[int, float], 
                         extracted_texts: Set[str], 
                         track_colors: Dict[int, Tuple[int, int, int]],
                         model: YOLO) -> None:
        """Process a single detection with OCR and visualization."""
        track_id = detection_data['track_id']
        cls_id = detection_data['cls_id']
        
        # Check if OCR is needed
        current_confidence = track_ocr_confidence.get(track_id, 0.0)
        should_attempt_ocr = current_confidence < self.config.confidence_threshold
        
        if should_attempt_ocr:
            self._handle_ocr_processing(
                frame, detection_data, track_id, cls_id, model,
                track_ocr_confidence, extracted_texts, track_colors
            )
        
        # Always draw visualization
        self._draw_detection(frame, detection_data, track_id, cls_id, model, track_colors)
    
    def _handle_ocr_processing(self, frame: np.ndarray, detection_data: dict,
                              track_id: int, cls_id: int, model: YOLO,
                              track_ocr_confidence: Dict[int, float],
                              extracted_texts: Set[str],
                              track_colors: Dict[int, Tuple[int, int, int]]) -> None:
        """Handle OCR processing for a detection."""
        current_confidence = track_ocr_confidence.get(track_id, 0.0)
        
        # Log detection status
        if track_id not in track_ocr_confidence:
            detection_type = "OBB" if detection_data['is_obb'] else "BBOX"
            print(f"\n--- NEW OBJECT DETECTED ({detection_type}) ---")
            print(f"  - Track ID: {track_id}, Class: {model.names[cls_id]}")
            track_colors[track_id] = (
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
        else:
            print(f"\n--- RETRYING OCR (Low Confidence: {current_confidence:.3f}) ---")
            print(f"  - Track ID: {track_id}, Class: {model.names[cls_id]}")
        
        # Extract crop and perform OCR
        crop_img = self._extract_crop(frame, detection_data)
        text, confidence = self.ocr_processor.extract_text_with_rotations(
            crop_img, self.config.ocr_rotations
        )
        
        # Update confidence and extracted texts
        if confidence > current_confidence:
            track_ocr_confidence[track_id] = confidence
            if text and text.strip():
                extracted_texts.add(text.strip())
            
            if confidence >= self.config.confidence_threshold:
                print(f"  - OCR SUCCESS: Confidence {confidence:.3f} meets threshold {self.config.confidence_threshold}")
            else:
                print(f"  - OCR RETRY NEEDED: Confidence {confidence:.3f} below threshold {self.config.confidence_threshold}")
        else:
            print(f"  - OCR: No improvement in confidence ({confidence:.3f} <= {current_confidence:.3f})")
    
    def _extract_crop(self, frame: np.ndarray, detection_data: dict) -> np.ndarray:
        """Extract crop based on detection type."""
        if detection_data['is_obb']:
            return self.crop_extractor.extract_obb_crop(frame, detection_data['coordinates'])
        else:
            return self.crop_extractor.extract_regular_crop(frame, detection_data['coordinates'])
    
    def _draw_detection(self, frame: np.ndarray, detection_data: dict,
                       track_id: int, cls_id: int, model: YOLO,
                       track_colors: Dict[int, Tuple[int, int, int]]) -> None:
        """Draw detection visualization."""
        color = track_colors.get(track_id, (0, 255, 0))
        class_name = model.names[cls_id]
        
        if detection_data['is_obb']:
            self.visualizer.draw_obb(frame, detection_data['coordinates'], track_id, class_name, color)
        else:
            self.visualizer.draw_bbox(frame, detection_data['coordinates'], track_id, class_name, color)


class VideoTracker:
    """Main tracker class that orchestrates the tracking process."""
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.model = None
        self.cap = None
        self.ocr_processor = OCRProcessor()
        self.detection_processor = DetectionProcessor(self.ocr_processor, config)
        
        # Tracking state
        self.track_ocr_confidence: Dict[int, float] = {}
        self.extracted_texts: Set[str] = set()
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}
    
    def initialize(self) -> bool:
        """Initialize the tracker components."""
        try:
            self.model = YOLO(self.config.model_path)
            print(f"✓ YOLO model loaded: {self.config.model_path}")
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            return False
        
        try:
            self.cap = cv2.VideoCapture(self.config.video_path)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file: {self.config.video_path}")
            print(f"✓ Video opened: {self.config.video_path}")
        except Exception as e:
            print(f"✗ Error opening video file: {e}")
            return False
        
        return True
    
    def run(self) -> None:
        """Run the main tracking loop."""
        if not self.initialize():
            return
        
        print("Successfully initialized. Starting tracker...")
        print(f"OCR Confidence Threshold: {self.config.confidence_threshold}")
        print("Press 'q' to quit.")
        
        frame_count = 0
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            frame_count += 1
            self._process_frame(frame)
            
            cv2.imshow("YOLO Object Tracker with OCR", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self._cleanup()
        self._print_summary()
    
    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame."""
        # Resize frame for performance
        if self.config.frame_resize_factor != 1.0:
            frame = cv2.resize(
                frame, None, 
                fx=self.config.frame_resize_factor, 
                fy=self.config.frame_resize_factor, 
                interpolation=cv2.INTER_AREA
            )
        
        # Run YOLO tracking
        results = self.model.track(frame, persist=True, tracker=self.config.tracker_config)
        
        if not results or not results[0]:
            return
        
        result = results[0]
        
        # Process detections based on type
        if hasattr(result, 'obb') and result.obb is not None and result.obb.id is not None:
            self._process_obb_detections(frame, result)
        elif hasattr(result, 'boxes') and result.boxes is not None and result.boxes.id is not None:
            self._process_regular_detections(frame, result)
    
    def _process_obb_detections(self, frame: np.ndarray, result) -> None:
        """Process OBB detections."""
        obb_coords = result.obb.xyxyxyxy.cpu().numpy()
        track_ids = result.obb.id.cpu().numpy().astype(int)
        class_ids = result.obb.cls.cpu().numpy().astype(int)
        
        for obb_points, track_id, cls_id in zip(obb_coords, track_ids, class_ids):
            detection_data = {
                'coordinates': obb_points,
                'track_id': track_id,
                'cls_id': cls_id,
                'is_obb': True
            }
            
            self.detection_processor.process_detection(
                frame, detection_data, self.track_ocr_confidence,
                self.extracted_texts, self.track_colors, self.model
            )
    
    def _process_regular_detections(self, frame: np.ndarray, result) -> None:
        """Process regular bounding box detections."""
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            detection_data = {
                'coordinates': box,
                'track_id': track_id,
                'cls_id': cls_id,
                'is_obb': False
            }
            
            self.detection_processor.process_detection(
                frame, detection_data, self.track_ocr_confidence,
                self.extracted_texts, self.track_colors, self.model
            )
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _print_summary(self) -> None:
        """Print final summary."""
        print("\n" + "="*50)
        print("TRACKER FINISHED")
        print("="*50)
        print("All unique texts found during the session:")
        
        if self.extracted_texts:
            for i, text in enumerate(sorted(self.extracted_texts), 1):
                print(f"{i:2d}: {text}")
        else:
            print("No unique text was extracted.")
        
        print(f"\nTotal unique texts: {len(self.extracted_texts)}")
        print(f"Total tracked objects: {len(self.track_ocr_confidence)}")


def run_tracker(video_path: str, model_path: str, tracker_config: str, 
                confidence_threshold: float = 0.7) -> None:
    """
    Main function to run the optimized YOLO object tracker.
    
    Args:
        video_path: Path to the video file
        model_path: Path to the YOLO model
        tracker_config: Tracker configuration file
        confidence_threshold: OCR confidence threshold
    """
    config = TrackerConfig(
        video_path=video_path,
        model_path=model_path,
        tracker_config=tracker_config,
        confidence_threshold=confidence_threshold
    )
    
    tracker = VideoTracker(config)
    tracker.run()


if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "data/5673626-hd_1920_1080_30fps.mp4"
    MODEL_PATH = 'yolo11n-obb.pt'
    TRACKER_CONFIG = 'bytetrack.yaml'
    CONFIDENCE_THRESHOLD = 0.7
    
    run_tracker(VIDEO_PATH, MODEL_PATH, TRACKER_CONFIG, CONFIDENCE_THRESHOLD)
