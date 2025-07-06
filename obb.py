from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model

# Predict with the model
# results = model.track("https://www.youtube.com/watch?v=n-D10TxEQdE", save=True, show=True, tracker="bytetrack.yaml")  # predict on an image

print("Starting video tracking...")
results = model.track("video_test.mp4", stream=True)  # with ByteTrack (removed show=True and tracker)
print("model.track() called. Entering results loop...")

# Initialize video writer
video_writer = None
output_filename = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
fps = 30 # Assuming 30 frames per second, adjust if needed

frame_count = 0

# Process results generator
for result in results:
    frame_count += 1
    print(f"Processing frame {frame_count}...")
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs

    # Get the annotated frame
    annotated_frame = result.plot()
    print(f"Annotated frame shape: {annotated_frame.shape}")

    # Initialize video writer on the first frame
    if video_writer is None:
        height, width, _ = annotated_frame.shape
        print(f"Initializing VideoWriter with size: ({width}, {height})")
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print("Error: VideoWriter not opened.")
            break # Exit loop if writer can't be opened

    # Write the frame to the video file
    video_writer.write(annotated_frame)
    print("Frame written to video file.")

print("Exited results loop.")

# Release the video writer after the loop
if video_writer is not None:
    print("Releasing VideoWriter...")
    video_writer.release()
    print("VideoWriter released.")

cv2.destroyAllWindows() # Close display windows after processing
print("Script finished.")

# # # Access the results
# for result in results:
#     xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
#     xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
#     names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
#     confs = result.obb.conf  # confidence score of each box