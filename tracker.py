import cv2
from ultralytics import YOLO
import random
from PIL import Image
import easyocr

def initialize_tracker(video_path, model_path, tracker_config):
    """
    Initializes the YOLO model, video capture, and other configurations.

    Args:
        video_path (str): Path to the video file.
        model_path (str): Path to the YOLO model.
        tracker_config (str): Path to the tracker configuration file.

    Returns:
        tuple: A tuple containing the YOLO model, video capture object,
               a set for processed track IDs, a set for extracted texts,
               and a dictionary for track colors.
    """
    # Initialize the EasyOCR reader.
    reader = easyocr.Reader(['en'])

    # A set to store the IDs of objects that have already been processed for OCR.
    processed_track_ids = set()
    
    # A set to store all unique texts extracted via OCR.
    extracted_texts = set()

    # A dictionary to store a unique color for each track ID.
    track_colors = {}

    # Initialize the YOLO model.
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None, None, None, None, None

    # Open the video file.
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
    except Exception as e:
        print(f"Error opening video file: {e}")
        return model, None, None, None, None, reader

    return model, cap, processed_track_ids, extracted_texts, track_colors, reader

def process_frame(frame, model, processed_track_ids, extracted_texts, track_colors, reader_instance, tracker_config):
    """
    Processes a single frame from the video, performing object detection, OCR, and visualization.

    Args:
        frame (numpy.ndarray): The video frame to process.
        model (YOLO): The YOLO model.
        processed_track_ids (set): Set of already processed track IDs.
        extracted_texts (set): Set of extracted texts.
        track_colors (dict): Dictionary of track colors.
        reader (easyocr.Reader): EasyOCR reader instance.
        tracker_config (str): Path to the tracker configuration file.
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

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            # --- NEW OBJECT DETECTION & OCR LOGIC ---
            if track_id not in processed_track_ids:
                print(f"\n--- NEW OBJECT DETECTED ---")
                print(f"  - Track ID: {track_id}, Class: {model.names[cls_id]}")

                # Add the new track_id to our set of processed IDs.
                processed_track_ids.add(track_id)
                # Assign a random color for the bounding box.
                track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # --- TESSERACT OCR INTEGRATION ---
                try:
                    # Crop the object from the resized frame using the bounding box coordinates.
                    x1, y1, x2, y2 = box
                    # Add a small buffer to ensure the whole object is captured.
                    crop_img = frame[max(0, y1):y2, max(0, x1):x2]

                    # Convert the cropped image (OpenCV BGR format) to a PIL Image (RGB format).
                    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

                    # Use EasyOCR to extract text.
                    ocr_result = reader_instance.readtext(crop_img)

                    if ocr_result:
                        text = ocr_result[0][1]  # Extract the text from the result
                        print(f"  - OCR Result: '{text}'")
                        # Add the cleaned text to our set of unique texts.
                        extracted_texts.add(text)
                    else:
                        print("  - OCR Result: No text found.")

                except Exception as ocr_error:
                    print(f"  - OCR Error: {ocr_error}")
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
    """
    model, cap, processed_track_ids, extracted_texts, track_colors, reader = initialize_tracker(video_path, model_path, tracker_config)

    if cap is None:
        return

    print("Successfully opened video. Starting tracker...")
    print("Press 'q' to quit.")

    # Main loop to process each frame of the video.
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        process_frame(frame, model, processed_track_ids, extracted_texts, track_colors, reader, tracker_config)

        cv2.imshow("YOLO Object Tracker with OCR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- FINAL SUMMARY ---
    cap.release()
    cv2.destroyAllWindows()
    generate_final_summary(extracted_texts)


if __name__ == "__main__":
    # --- CONFIGURATION ---

    # For Windows users, you might need to specify the path to the Tesseract executable.

    # Path to the video file to be processed.
    # A video with clear text (like street signs, license plates, or book spines) is ideal for testing.
    VIDEO_PATH = "data/5673626-hd_1920_1080_30fps.mp4" 

    # Path to the YOLO model file. 'yolov8n.pt' is a small, fast model.
    MODEL_PATH = 'license_plate_detector.pt'

    # Tracker configuration file. ByteTrack is a robust tracker.
    # Other options include 'botsort.yaml'.
    TRACKER_CONFIG = 'bytetrack.yaml'
    run_tracker(VIDEO_PATH, MODEL_PATH, TRACKER_CONFIG)
