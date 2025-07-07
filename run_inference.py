from ultralytics import YOLO
import cv2
import numpy as np
from ocr_processor import OCRProcessor
from detection_processor import DetectionProcessor
from config import TrackerConfig

# Path to the trained model weights
model_path = 'best_custom_trained_yolo-obb.pt'

# Path to the image for inference
image_path = 'data/test_image/2b47d185775e961f05daa18fd11f43e5_jpg.rf.be74949a2e706b6e73098392510a1b43.jpg'

def run_inference():
    """
    Performs single-image inference using a pre-trained YOLO OBB model.
    It detects objects, extracts oriented bounding box (OBB) coordinates,
    and then applies OCR (Optical Character Recognition) to the detected regions.
    The processed image with OBB detections and OCR results is saved.
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Running inference on the single image: {image_path}")
    # The 'save=True' argument saves the annotated image with OBB detections
    # to a directory like 'runs/detect/predict' or 'runs/detect/predict2', etc.
    results = model.predict(source=image_path, save=True, conf=0.25)

    # Load the image for further processing, including OCR
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Initialize OCRProcessor and DetectionProcessor
    ocr_processor = OCRProcessor()
    # Configure DetectionProcessor for single-image OCR.
    # ocr_frame_interval=0 and movement_threshold=0.0 ensure OCR is always attempted
    # for detected objects in this single frame context, bypassing typical video optimization.
    config = TrackerConfig(
        video_path="dummy_video.mp4", # Not used for single image, but required by TrackerConfig
        model_path=model_path,
        tracker_config="dummy_tracker_config.yaml", # Not used for single image, but required
        ocr_frame_interval=0, # Force OCR on every detection in this single frame
        movement_threshold=0.0 # Ignore movement for OCR triggering in single frame
    )
    detection_processor = DetectionProcessor(ocr_processor, config)

    # Dictionaries to hold OCR results and other tracking data (even for single image)
    track_ocr_confidence = {}
    extracted_texts = set()
    track_colors = {}
    frame_count = 0 # For single image inference, frame_count is effectively 0 or 1

    print("\nInference complete. Processing detections and performing OCR.")
    for r in results:
        if r.obb is not None:
            print(f"Found {len(r.obb)} OBB detections. Applying OCR...")
            for i, obb_coords in enumerate(r.obb.xyxyxyxy):
                # Convert OBB coordinates to a numpy array for processing
                obb_coords_np = np.array(obb_coords).reshape(-1, 2)
                
                # Assign a dummy track_id for each detection as no actual tracking is involved
                track_id = i 
                cls_id = int(r.obb.cls[i]) # Class ID of the detected object
                
                detection_data = {
                    'track_id': track_id,
                    'cls_id': cls_id,
                    'coordinates': obb_coords_np, # Oriented Bounding Box coordinates
                    'is_obb': True # Indicate that these are OBB detections
                }
                
                # Process each detection: crop the OBB region and perform OCR
                detection_processor.process_detection(
                    frame, detection_data, track_ocr_confidence, 
                    extracted_texts, track_colors, model, frame_count
                )
        else:
            print("No OBB detections found in the image.")
        
        # Save the processed image, which now includes OBB visualizations and OCR results
        # (DetectionProcessor draws directly onto the 'frame' object).
        output_image_path = "runs/processed_inference_image.jpg"
        cv2.imwrite(output_image_path, frame)
        print(f"Processed image with OBB detections and OCR results saved to: {output_image_path}")
        
        print("\n--- Extracted Texts from OCR ---")
        if extracted_texts:
            for text in extracted_texts:
                print(f"- {text}")
        else:
            print("No text was extracted by OCR from the detected objects.")

    print("\nSingle image inference and OCR processing complete.")

if __name__ == "__main__":
    run_inference()
