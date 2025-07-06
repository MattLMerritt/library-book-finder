from ultralytics import YOLO
import random
import ultralytics

# Hyperparameters section
epochs = 50  # Reduced number of epochs
batch_size = 16
learning_rate = 0.001

# Load a model
model = YOLO("yolo11n-obb.pt")  # load a pretrained model

# Load dataset configuration
data_yaml = "data/LibVision 2 -No Other Class-.v1i.yolov8-obb/data.yaml"

# Limit training data (example: use 20% of the data)
subset_ratio = 0.001

# Train the model
results = model.train(
    data=data_yaml,
    epochs=epochs,
    batch=batch_size,
    lr0=learning_rate,
    fraction=subset_ratio # Use a subset of the data for faster training
)

# Evaluate the model's performance on the validation set
metrics = model.val()  # evaluate model performance on the validation set

# Reset unsupported arguments before exporting
model.args.pop("fraction", None)  # Remove 'fraction' if it exists in model arguments

# Export the trained model to a format suitable for deployment
model.export(format="onnx")  # export the model to ONNX format
