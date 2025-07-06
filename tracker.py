"""
Optimized YOLO Object Tracker with OCR
Supports both regular bounding boxes and oriented bounding boxes (OBB).
Features confidence-based OCR retry system for improved text extraction accuracy.
PERFORMANCE OPTIMIZATIONS:
- Frame interval OCR processing to prevent repeated processing
- Pre-filtering of poor OCR candidates
- Smart rotation strategy with early exit
- Frame resizing before YOLO inference
- Object movement tracking to skip static objects
"""

import cv2
import numpy as np
import time
import heapq # Import for priority queue
from typing import Dict, Set, Tuple, Optional, Union, List
from ultralytics import YOLO

# Import refactored modules
from config import TrackerConfig
from data_models import DetectionPriority, ObjectTrackingState
from ocr_processor import OCRProcessor
from crop_extractor import CropExtractor
from visualizer import Visualizer
from detection_processor import DetectionProcessor


class VideoTracker:
    """Main tracker class that orchestrates the tracking process with performance optimizations."""
    
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
        self.frame_count = 0
    
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
        """Run the main tracking loop with performance optimizations."""
        if not self.initialize():
            return
        
        print("Successfully initialized. Starting optimized tracker...")
        print(f"OCR Confidence Threshold: {self.config.confidence_threshold}")
        print(f"OCR Frame Interval: {self.config.ocr_frame_interval}")
        print(f"Frame Resize Factor: {self.config.frame_resize_factor}")
        print("Press 'q' to quit.")
        
        start_time = time.time()
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            self.frame_count += 1
            self._process_frame(frame)
            
            cv2.imshow("Optimized YOLO Object Tracker with OCR", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        end_time = time.time()
        self._cleanup()
        self._print_summary(end_time - start_time)
    
    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame with optimizations."""
        # OPTIMIZATION: Resize frame BEFORE YOLO inference for better performance
        original_frame = frame.copy()
        if self.config.frame_resize_factor != 1.0:
            frame = cv2.resize(
                frame, None, 
                fx=self.config.frame_resize_factor, 
                fy=self.config.frame_resize_factor, 
                interpolation=cv2.INTER_AREA
            )
        
        # Run YOLO tracking on resized frame
        results = self.model.track(frame, persist=True, tracker=self.config.tracker_config)
        
        if not results or not results[0]:
            return
        
        result = results[0]
        
        # Scale coordinates back to original frame size if needed
        scale_factor = 1.0 / self.config.frame_resize_factor if self.config.frame_resize_factor != 1.0 else 1.0
        
        # Collect all detections with their priorities
        all_detections_with_priority: List[DetectionPriority] = []

        if hasattr(result, 'obb') and result.obb is not None and result.obb.id is not None:
            obb_coords = result.obb.xyxyxyxy.cpu().numpy()
            track_ids = result.obb.id.cpu().numpy().astype(int)
            class_ids = result.obb.cls.cpu().numpy().astype(int)
            confidences = result.obb.conf.cpu().numpy()

            for obb_points, track_id, cls_id, conf in zip(obb_coords, track_ids, class_ids, confidences):
                if scale_factor != 1.0:
                    obb_points = obb_points * scale_factor
                
                detection_data = {
                    'coordinates': obb_points,
                    'track_id': track_id,
                    'cls_id': cls_id,
                    'is_obb': True,
                    'confidence': conf
                }
                priority = self._calculate_detection_priority(detection_data, conf)
                heapq.heappush(all_detections_with_priority, DetectionPriority(
                    priority=priority,
                    frame_count=self.frame_count,
                    detection_data=detection_data,
                    track_id=track_id,
                    cls_id=cls_id,
                    model=self.model
                ))

        if hasattr(result, 'boxes') and result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                if scale_factor != 1.0:
                    box = (box * scale_factor).astype(int)
                
                detection_data = {
                    'coordinates': box,
                    'track_id': track_id,
                    'cls_id': cls_id,
                    'is_obb': False,
                    'confidence': conf
                }
                priority = self._calculate_detection_priority(detection_data, conf)
                heapq.heappush(all_detections_with_priority, DetectionPriority(
                    priority=priority,
                    frame_count=self.frame_count,
                    detection_data=detection_data,
                    track_id=track_id,
                    cls_id=cls_id,
                    model=self.model
                ))
        
        # Process detections by priority
        while all_detections_with_priority:
            detection_prio_obj = heapq.heappop(all_detections_with_priority)
            self.detection_processor.process_detection(
                original_frame, 
                detection_prio_obj.detection_data, 
                self.track_ocr_confidence,
                self.extracted_texts, 
                self.track_colors, 
                detection_prio_obj.model, 
                detection_prio_obj.frame_count
            )
    
    def _calculate_detection_priority(self, detection_data: Dict, confidence: float) -> float:
        """
        Calculate a priority score for a detection. Higher score means higher priority.
        Factors: confidence, size (area), whether it's an OBB.
        """
        # Base priority is confidence
        priority = confidence

        # Add bonus for larger objects (more likely to have readable text)
        coords = detection_data['coordinates']
        if detection_data['is_obb']:
            # For OBB, calculate area from minAreaRect
            points = coords.reshape(4, 2).astype(np.float32)
            rect = cv2.minAreaRect(points)
            area = rect[1][0] * rect[1][1]
        else:
            # For regular bbox, calculate area from x1,y1,x2,y2
            x1, y1, x2, y2 = coords
            area = (x2 - x1) * (y2 - y1)
        
        # Normalize area (example: max area of 10000 pixels gives +1.0 bonus)
        # Adjust this normalization factor based on expected object sizes
        area_normalized = min(area / 10000.0, 1.0) 
        priority += area_normalized * 0.5 # Add up to 0.5 bonus for large objects

        # Add a small bonus for OBB detections if they are generally more accurate for text
        if detection_data['is_obb']:
            priority += 0.1

        # Invert priority for min-heap (heapq) to act as max-heap
        return -priority

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _print_summary(self, total_time: float) -> None:
        """Print final summary with performance metrics."""
        print("\n" + "="*60)
        print("OPTIMIZED TRACKER FINISHED")
        print("="*60)
        print("All unique texts found during the session:")
        
        if self.extracted_texts:
            for i, text in enumerate(sorted(self.extracted_texts), 1):
                print(f"{i:2d}: {text}")
        else:
            print("No unique text was extracted.")
        
        print(f"\nPerformance Summary:")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average FPS: {self.frame_count / total_time:.2f}")
        print(f"Total unique texts: {len(self.extracted_texts)}")
        print(f"Total tracked objects: {len(self.track_ocr_confidence)}")
        
        # OCR attempt statistics
        total_ocr_attempts = sum(state.ocr_attempts for state in self.detection_processor.object_states.values())
        print(f"Total OCR attempts: {total_ocr_attempts}")
        print(f"Average OCR attempts per object: {total_ocr_attempts / max(1, len(self.track_ocr_confidence)):.1f}")


def run_tracker(video_path: str, model_path: str, tracker_config: str, 
                confidence_threshold: float = 0.7, frame_resize_factor: float = 0.5,
                ocr_frame_interval: int = 5) -> None:
    """
    Main function to run the optimized YOLO object tracker.
    
    Args:
        video_path: Path to the video file
        model_path: Path to the YOLO model
        tracker_config: Tracker configuration file
        confidence_threshold: OCR confidence threshold
        frame_resize_factor: Factor to resize frames for performance
        ocr_frame_interval: Only attempt OCR every N frames per object
    """
    config = TrackerConfig(
        video_path=video_path,
        model_path=model_path,
        tracker_config=tracker_config,
        confidence_threshold=confidence_threshold,
        frame_resize_factor=frame_resize_factor,
        ocr_frame_interval=ocr_frame_interval
    )
    
    tracker = VideoTracker(config)
    tracker.run()


if __name__ == "__main__":
    # Configuration with optimized defaults
    VIDEO_PATH = "data/5673626-hd_1920_1080_30fps.mp4"
    MODEL_PATH = 'yolo11n-obb.pt'
    TRACKER_CONFIG = 'bytetrack.yaml'
    CONFIDENCE_THRESHOLD = 0.7
    FRAME_RESIZE_FACTOR = 0.5  # Resize frames to 50% for better performance
    OCR_FRAME_INTERVAL = 15  # Only attempt OCR every 5 frames per object
    
    run_tracker(VIDEO_PATH, MODEL_PATH, TRACKER_CONFIG, CONFIDENCE_THRESHOLD, 
                FRAME_RESIZE_FACTOR, OCR_FRAME_INTERVAL)
