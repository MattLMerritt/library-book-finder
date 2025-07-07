from ultralytics import YOLO

# Path to the trained model weights
model_path = 'best_custom_trained_yolo-obb.pt'

# Path to the image for inference
image_path = 'data/test_image/2b47d185775e961f05daa18fd11f43e5_jpg.rf.be74949a2e706b6e73098392510a1b43.jpg'

def run_inference():
    """
    Loads a YOLO model and performs inference on a specified image.
    Results (annotated image, etc.) will be saved in runs/detect/predictX.
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Running inference on image: {image_path}")
    # The 'save=True' argument saves the annotated image and other results
    # to a directory like 'runs/detect/predict' or 'runs/detect/predict2', etc.
    results = model.predict(source=image_path, save=True, conf=0.25)

    print("\nInference complete. Results saved to 'runs/obb/predict' directory.")
    for r in results:
        if r.boxes is not None:
            print(f"Boxes: {len(r.boxes)} detected")
        else:
            print("No boxes detected.")
        if r.obb is not None:
            print(f"OBB: {len(r.obb)} detected")
        else:
            print("No OBB detected.")
        if r.masks is not None:
            print(f"Masks: {len(r.masks)} detected")
        else:
            print("No masks detected.")
        if r.keypoints is not None:
            print(f"Keypoints: {len(r.keypoints)} detected")
        else:
            print("No keypoints detected.")
        if r.probs is not None:
            print(f"Probs: {len(r.probs)} detected")
        else:
            print("No probabilities detected.")

if __name__ == "__main__":
    run_inference()
