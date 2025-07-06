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
import random
import hashlib
import time
import heapq # Import for priority queue
from typing import Dict, Set, Tuple, Optional, Union, List
from dataclasses import dataclass, field
from ultralytics import YOLO
import easyocr


@dataclass(order=True)
class DetectionPriority:
    """
    Helper class for priority queue.
    Priority is negative so that heapq (min-heap) can be used as a max-heap.
    """
    priority: float
    frame_count: int = field(compare=False)
    detection_data: Dict = field(compare=False)
    track_id: int = field(compare=False)
    cls_id: int = field(compare=False)
    model: YOLO = field(compare=False)


@dataclass
class TrackerConfig:
    """Configuration class for tracker parameters."""
    video_path: str
    model_path: str
    tracker_config: str
    confidence_threshold: float = 0.7
    frame_resize_factor: float = 0.5
    min_crop_size: int = 20
    ocr_rotations: Tuple[int, ...] = (90, 180, 270)
    # Performance optimization parameters
    ocr_frame_interval: int = 5  # Only attempt OCR every N frames per object
    min_crop_area: int = 400  # Minimum crop area for OCR
    max_aspect_ratio: float = 10.0  # Skip crops with extreme aspect ratios
    min_contrast_threshold: float = 20.0  # Minimum contrast for OCR
    movement_threshold: float = 10.0  # Minimum movement to trigger new OCR
    early_exit_confidence: float = 0.9  # Stop rotations if this confidence is reached


@dataclass
class ObjectTrackingState:
    """Tracks state for individual objects to optimize processing."""
    last_ocr_frame: int = 0
    last_position: Optional[Tuple[float, float]] = None
    ocr_attempts: int = 0
    crop_hash_cache: Dict[str, Tuple[str, float]] = field(default_factory=dict)


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
    OCR_FRAME_INTERVAL = 5  # Only attempt OCR every 5 frames per object
    
    run_tracker(VIDEO_PATH, MODEL_PATH, TRACKER_CONFIG, CONFIDENCE_THRESHOLD, 
                FRAME_RESIZE_FACTOR, OCR_FRAME_INTERVAL)
