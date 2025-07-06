from dataclasses import dataclass
from typing import Tuple

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
