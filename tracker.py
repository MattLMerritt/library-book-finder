import cv2
from ultralytics import YOLO
import random
from PIL import Image
import easyocr
import numpy as np

def initialize_tracker(video_path, model_path, tracker_config, confidence_threshold=0.7):
    """
    Initializes the YOLO model, video capture, and other configurations.

    Args:
        video_path (str): Path to the video file.
        model_path (str): Path to the YOLO model.
        tracker_config (str): Path to the tracker configuration file.
        confidence_threshold (float): Minimum confidence threshold for OCR success (0.0-1.0).

    Returns:
        tuple: A tuple containing the YOLO model, video capture object,
               tracking data structures, and configuration values.
    """
    # Initialize the EasyOCR reader.
    reader = easyocr.Reader(['en'])

    # A dictionary to store OCR confidence scores for each track ID.
    track_ocr_confidence = {}
    
    # A set to store all unique texts extracted via OCR.
    extracted_texts = set()

    # A dictionary to store a unique color for each track ID.
    track_colors = {}

    # Initialize the YOLO model.
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None, None, None, None, None, None

    # Open the video file.
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
    except Exception as e:
        print(f"Error opening video file: {e}")
        return model, None, None, None, None, reader, confidence_threshold

    return model, cap, track_ocr_confidence, extracted_texts, track_colors, reader, confidence_threshold

def perform_ocr_on_crop(crop_img, reader_instance, extracted_texts):
    """
    Performs OCR on a cropped image and adds extracted text to the collection.
    Enhanced for OBB models with rotation handling and multiple orientation attempts.

    Args:
        crop_img (numpy.ndarray): The cropped image containing the object.
        reader_instance (easyocr.Reader): EasyOCR reader instance.
        extracted_texts (set): Set to store extracted texts.

    Returns:
        tuple: (text, confidence) where text is the extracted text and confidence is the OCR confidence score (0.0-1.0).
               Returns (None, 0.0) if no text found or error occurred.
    """
    try:
        # Skip if crop is too small
        if crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
            print("  - OCR Result: Crop too small, skipping.")
            return None, 0.0

        best_text = None
        best_confidence = 0.0
        
        # Try OCR on original orientation first
        ocr_result = reader_instance.readtext(crop_img)
        
        if ocr_result:
            for detection in ocr_result:
                text = detection[1]
                confidence = detection[2]
                if confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence

        # For OBB models, also try different rotations to improve OCR accuracy
        # This is especially useful for books that might be at various angles
        rotations = [90, 180, 270]  # Try rotating the image
        
        for angle in rotations:
            try:
                # Rotate the crop
                if angle == 90:
                    rotated_crop = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated_crop = cv2.rotate(crop_img, cv2.ROTATE_180)
                elif angle == 270:
                    rotated_crop = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Try OCR on rotated image
                ocr_result_rotated = reader_instance.readtext(rotated_crop)
                
                if ocr_result_rotated:
                    for detection in ocr_result_rotated:
                        text = detection[1]
                        confidence = detection[2]
                        if confidence > best_confidence:
                            best_text = text
                            best_confidence = confidence
                            print(f"  - Better OCR found at {angle}° rotation")
                            
            except Exception as rotation_error:
                # Continue if rotation fails
                continue

        if best_text and best_confidence > 0:
            print(f"  - OCR Result: '{best_text}' (Confidence: {best_confidence:.3f})")
            # Clean and add the text to our set of unique texts
            cleaned_text = best_text.strip()
            if cleaned_text:  # Only add non-empty text
                extracted_texts.add(cleaned_text)
            return best_text, best_confidence
        else:
            print("  - OCR Result: No text found.")
            return None, 0.0

    except Exception as ocr_error:
        print(f"  - OCR Error: {ocr_error}")
        return None, 0.0

def extract_rotated_crop(frame, obb_points):
    """
    Extracts a rotated crop from the frame using oriented bounding box points.
    
    Args:
        frame (numpy.ndarray): The input frame
        obb_points (numpy.ndarray): 8-point OBB coordinates [x1,y1,x2,y2,x3,y3,x4,y4]
    
    Returns:
        numpy.ndarray: Cropped and rotated image
    """
    try:
        # Reshape OBB points to 4x2 format
        points = obb_points.reshape(4, 2).astype(np.float32)
        
        # Calculate the bounding rectangle
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)  # Use np.int32 instead of deprecated np.int0
        
        # Get width and height of the rotated rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        # Ensure minimum size
        if width < 10 or height < 10:
            raise ValueError("Rotated crop too small")
        
        # Define destination points for perspective transform
        dst_pts = np.array([
            [0, height-1],
            [0, 0],
            [width-1, 0],
            [width-1, height-1]
        ], dtype=np.float32)
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)
        
        # Apply perspective transform
        rotated_crop = cv2.warpPerspective(frame, M, (width, height))
        
        return rotated_crop
        
    except Exception as e:
        print(f"Error extracting rotated crop: {e}")
        # Fallback to regular bounding box crop
        try:
            # Flatten and extract coordinates properly
            coords = obb_points.flatten()
            x_coords = coords[::2]  # x coordinates
            y_coords = coords[1::2]  # y coordinates
            
            x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
            y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))
            
            # Ensure valid crop bounds
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]
            else:
                # Return a small dummy crop if bounds are invalid
                return np.zeros((20, 20, 3), dtype=np.uint8)
                
        except Exception as fallback_error:
            print(f"Fallback crop also failed: {fallback_error}")
            # Return a small dummy crop as last resort
            return np.zeros((20, 20, 3), dtype=np.uint8)

def process_frame(frame, model, track_ocr_confidence, extracted_texts, track_colors, reader_instance, tracker_config, confidence_threshold):
    """
    Processes a single frame from the video, performing object detection, OCR, and visualization.
    Handles both regular bounding boxes and oriented bounding boxes (OBB).

    Args:
        frame (numpy.ndarray): The video frame to process.
        model (YOLO): The YOLO model.
        track_ocr_confidence (dict): Dictionary storing OCR confidence scores for each track ID.
        extracted_texts (set): Set of extracted texts.
        track_colors (dict): Dictionary of track colors.
        reader_instance (easyocr.Reader): EasyOCR reader instance.
        tracker_config (str): Path to the tracker configuration file.
        confidence_threshold (float): Minimum confidence threshold for OCR success.
    """
    # --- PREPROCESSING: Resize the frame ---
    try:
        # Resize the frame to a smaller resolution (e.g., 50% of the original size).
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    except Exception as resize_error:
        print(f"Error resizing frame: {resize_error}")
        return

    # Run YOLO tracking on the resized frame.
    results = model.track(frame, persist=True, tracker=tracker_config)

    # Check if we have OBB results or regular box results
    has_obb = hasattr(results[0], 'obb') and results[0].obb is not None
    has_boxes = hasattr(results[0], 'boxes') and results[0].boxes is not None and results[0].boxes.id is not None

    if has_obb and results[0].obb.id is not None:
        # Handle OBB (Oriented Bounding Box) format
        obb_coords = results[0].obb.xyxyxyxy.cpu().numpy()  # 8-point coordinates
        track_ids = results[0].obb.id.cpu().numpy().astype(int)
        class_ids = results[0].obb.cls.cpu().numpy().astype(int)

        for obb_points, track_id, cls_id in zip(obb_coords, track_ids, class_ids):
            # --- CONFIDENCE-BASED OCR LOGIC ---
            current_confidence = track_ocr_confidence.get(track_id, 0.0)
            should_attempt_ocr = current_confidence < confidence_threshold
            
            if should_attempt_ocr:
                if track_id not in track_ocr_confidence:
                    print(f"\n--- NEW OBJECT DETECTED (OBB) ---")
                    print(f"  - Track ID: {track_id}, Class: {model.names[cls_id]}")
                    # Assign a random color for the bounding box.
                    track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    print(f"\n--- RETRYING OCR (Low Confidence: {current_confidence:.3f}) ---")
                    print(f"  - Track ID: {track_id}, Class: {model.names[cls_id]}")

                # --- OCR PROCESSING ---
                # Extract rotated crop using OBB coordinates
                crop_img = extract_rotated_crop(frame, obb_points)
                
                # Perform OCR on the cropped image
                text, confidence = perform_ocr_on_crop(crop_img, reader_instance, extracted_texts)
                
                # Update the confidence score for this track ID
                if confidence > current_confidence:
                    track_ocr_confidence[track_id] = confidence
                    if confidence >= confidence_threshold:
                        print(f"  - OCR SUCCESS: Confidence {confidence:.3f} meets threshold {confidence_threshold}")
                    else:
                        print(f"  - OCR RETRY NEEDED: Confidence {confidence:.3f} below threshold {confidence_threshold}")
                else:
                    print(f"  - OCR: No improvement in confidence ({confidence:.3f} <= {current_confidence:.3f})")
                # ---------------------------------

            # --- VISUALIZATION ---
            # Draw oriented bounding box
            points = obb_points.reshape(4, 2).astype(int)
            color = track_colors.get(track_id, (0, 255, 0))
            cv2.polylines(frame, [points], True, color, 2)
            
            # Add label
            label = f"ID:{track_id} {model.names[cls_id]}"
            label_pos = (int(points[0][0]), int(points[0][1]) - 10)
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (label_pos[0], label_pos[1] - text_height - 5), 
                         (label_pos[0] + text_width, label_pos[1]), color, -1)
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif has_boxes:
        # Handle regular bounding box format (fallback)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            # --- CONFIDENCE-BASED OCR LOGIC ---
            current_confidence = track_ocr_confidence.get(track_id, 0.0)
            should_attempt_ocr = current_confidence < confidence_threshold
            
            if should_attempt_ocr:
                if track_id not in track_ocr_confidence:
                    print(f"\n--- NEW OBJECT DETECTED (BBOX) ---")
                    print(f"  - Track ID: {track_id}, Class: {model.names[cls_id]}")
                    # Assign a random color for the bounding box.
                    track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    print(f"\n--- RETRYING OCR (Low Confidence: {current_confidence:.3f}) ---")
                    print(f"  - Track ID: {track_id}, Class: {model.names[cls_id]}")

                # --- OCR PROCESSING ---
                # Crop the object from the resized frame using the bounding box coordinates.
                x1, y1, x2, y2 = box
                # Add a small buffer to ensure the whole object is captured.
                crop_img = frame[max(0, y1):y2, max(0, x1):x2]
                
                # Perform OCR on the cropped image
                text, confidence = perform_ocr_on_crop(crop_img, reader_instance, extracted_texts)
                
                # Update the confidence score for this track ID
                if confidence > current_confidence:
                    track_ocr_confidence[track_id] = confidence
                    if confidence >= confidence_threshold:
                        print(f"  - OCR SUCCESS: Confidence {confidence:.3f} meets threshold {confidence_threshold}")
                    else:
                        print(f"  - OCR RETRY NEEDED: Confidence {confidence:.3f} below threshold {confidence_threshold}")
                else:
                    print(f"  - OCR: No improvement in confidence ({confidence:.3f} <= {current_confidence:.3f})")
                # ---------------------------------

            # --- VISUALIZATION ---
            x1, y1, x2, y2 = box
            color = track_colors.get(track_id, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {model.names[cls_id]}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # cv2.imshow("YOLO Object Tracker with OCR", frame)


def generate_final_summary(extracted_texts):
    """
    Generates and prints the final summary of the tracking session.

    Args:
        extracted_texts (set): Set of extracted texts.
    """
    print("\n--- TRACKER FINISHED ---")
    print("All unique texts found during the session:")
    if extracted_texts:
        for i, text in enumerate(extracted_texts):
            print(f"{i + 1}: {text}")
    else:
        print("No unique text was extracted.")

def run_tracker(video_path, model_path, tracker_config):
    """
    Initializes and runs the YOLO object tracker on a video file.
    It identifies new objects, performs OCR on them, and stores unique text.
    Uses confidence-based OCR retry system for improved accuracy.
    """
    model, cap, track_ocr_confidence, extracted_texts, track_colors, reader, confidence_threshold = initialize_tracker(video_path, model_path, tracker_config)

    if cap is None:
        return

    print("Successfully opened video. Starting tracker...")
    print(f"OCR Confidence Threshold: {confidence_threshold}")
    print("Press 'q' to quit.")

    # Main loop to process each frame of the video.
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        process_frame(frame, model, track_ocr_confidence, extracted_texts, track_colors, reader, tracker_config, confidence_threshold)

        cv2.imshow("YOLO Object Tracker with OCR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- FINAL SUMMARY ---
    cap.release()
    cv2.destroyAllWindows()
    generate_final_summary(extracted_texts)


if __name__ == "__main__":
    # --- CONFIGURATION ---

    # Path to the video file to be processed.
    # A video with clear text (like street signs, license plates, or book spines) is ideal for testing.
    VIDEO_PATH = "data/5673626-hd_1920_1080_30fps.mp4" 

    # Path to the YOLO OBB model file. Using the trained OBB model for oriented bounding boxes.
    MODEL_PATH = 'yolo11n-obb.pt'

    # Tracker configuration file. ByteTrack is a robust tracker.
    # Other options include 'botsort.yaml'.
    TRACKER_CONFIG = 'bytetrack.yaml'
    run_tracker(VIDEO_PATH, MODEL_PATH, TRACKER_CONFIG)
