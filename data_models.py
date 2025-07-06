from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

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
    model: object = field(compare=False) # Use object for YOLO to avoid circular import

@dataclass
class ObjectTrackingState:
    """Tracks state for individual objects to optimize processing."""
    last_ocr_frame: int = 0
    last_position: Optional[Tuple[float, float]] = None
    ocr_attempts: int = 0
    crop_hash_cache: Dict[str, Tuple[str, float]] = field(default_factory=dict)
