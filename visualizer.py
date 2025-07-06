import cv2
import numpy as np
from typing import Tuple

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
