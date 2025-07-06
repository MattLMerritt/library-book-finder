import cv2
import numpy as np
import random
from typing import Dict, Set, Tuple, Optional, Union, List
from ultralytics import YOLO # For type hinting model
from ocr_processor import OCRProcessor
from crop_extractor import CropExtractor
from visualizer import Visualizer
from config import TrackerConfig
from data_models import ObjectTrackingState # Import ObjectTrackingState

class DetectionProcessor:
    """Processes detections and manages OCR operations with performance optimizations."""
    
    def __init__(self, ocr_processor: OCRProcessor, config: TrackerConfig):
        self.ocr_processor = ocr_processor
        self.config = config
        self.crop_extractor = CropExtractor()
        self.visualizer = Visualizer()
        # Track object states for optimization
        self.object_states: Dict[int, ObjectTrackingState] = {}
    
    def process_detection(self, frame: np.ndarray, detection_data: dict, 
                         track_ocr_confidence: Dict[int, float], 
                         extracted_texts: Set[str], 
                         track_colors: Dict[int, Tuple[int, int, int]],
                         model: YOLO, frame_count: int) -> None:
        """Process a single detection with OCR and visualization."""
        track_id = detection_data['track_id']
        cls_id = detection_data['cls_id']
        
        # Initialize object state if new
        if track_id not in self.object_states:
            self.object_states[track_id] = ObjectTrackingState()
        
        # Check if OCR is needed with optimizations
        current_confidence = track_ocr_confidence.get(track_id, 0.0)
        should_attempt_ocr = self._should_attempt_ocr(
            track_id, current_confidence, detection_data, frame_count
        )
        
        if should_attempt_ocr:
            self._handle_ocr_processing(
                frame, detection_data, track_id, cls_id, model,
                track_ocr_confidence, extracted_texts, track_colors, frame_count
            )
        
        # Always draw visualization
        self._draw_detection(frame, detection_data, track_id, cls_id, model, track_colors)
    
    def _should_attempt_ocr(self, track_id: int, current_confidence: float, 
                           detection_data: dict, frame_count: int) -> bool:
        """Determine if OCR should be attempted based on various optimization criteria."""
        # Skip if confidence threshold already met
        if current_confidence >= self.config.confidence_threshold:
            return False
        
        obj_state = self.object_states[track_id]
        
        # Frame interval check - only attempt OCR every N frames
        frames_since_last_ocr = frame_count - obj_state.last_ocr_frame
        if frames_since_last_ocr < self.config.ocr_frame_interval:
            return False
        
        # Movement check - skip if object hasn't moved significantly
        current_pos = self._get_object_center(detection_data)
        if obj_state.last_position is not None:
            distance = np.sqrt(
                (current_pos[0] - obj_state.last_position[0])**2 + 
                (current_pos[1] - obj_state.last_position[1])**2
            )
            if distance < self.config.movement_threshold and obj_state.ocr_attempts > 0:
                return False
        
        return True
    
    def _get_object_center(self, detection_data: dict) -> Tuple[float, float]:
        """Get the center point of an object for movement tracking."""
        if detection_data['is_obb']:
            coords = detection_data['coordinates'].flatten()
            x_coords, y_coords = coords[::2], coords[1::2]
            return (np.mean(x_coords), np.mean(y_coords))
        else:
            x1, y1, x2, y2 = detection_data['coordinates']
            return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _handle_ocr_processing(self, frame: np.ndarray, detection_data: dict,
                              track_id: int, cls_id: int, model: YOLO,
                              track_ocr_confidence: Dict[int, float],
                              extracted_texts: Set[str],
                              track_colors: Dict[int, Tuple[int, int, int]],
                              frame_count: int) -> None:
        """Handle OCR processing for a detection with optimizations."""
        current_confidence = track_ocr_confidence.get(track_id, 0.0)
        obj_state = self.object_states[track_id]
        
        # Update tracking state
        obj_state.last_ocr_frame = frame_count
        obj_state.last_position = self._get_object_center(detection_data)
        obj_state.ocr_attempts += 1
        
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
            print(f"  - Track ID: {track_id}, Class: {model.names[cls_id]}, Attempt: {obj_state.ocr_attempts}")
        
        # Extract crop and perform OCR
        crop_img = self._extract_crop(frame, detection_data)
        
        # Pass obb_points to extract_text_with_rotations for OBB-aware OCR
        text, confidence = self.ocr_processor.extract_text_with_rotations(
            crop_img, detection_data['coordinates'], self.config.early_exit_confidence
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
