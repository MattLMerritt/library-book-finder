from ultralytics import YOLO
# Define multiple hyperparameter sets for experimentation
hyperparameter_sets = [
    {"epochs": 50, "batch_size": 16, "learning_rate": 0.001},
    {"epochs": 100, "batch_size": 16, "learning_rate": 0.001},
    {"epochs": 50, "batch_size": 32, "learning_rate": 0.0005},
    {"epochs": 100, "batch_size": 32, "learning_rate": 0.0005},
]

# Load dataset configuration
data_yaml = "data/book_only.yaml"

for i, hp_set in enumerate(hyperparameter_sets):
    epochs = hp_set["epochs"]
    batch_size = hp_set["batch_size"]
    learning_rate = hp_set["learning_rate"]

    print(f"\n--- Training with Hyperparameter Set {i+1} ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    # Load a fresh model for each training run
    model = YOLO("yolo11n-obb.pt")  # load a pretrained model

    # Reconfigure the model's head for 1 class
    if hasattr(model.model, 'model') and len(model.model.model) > 0 and hasattr(model.model.model[-1], 'nc'):
        model.model.model[-1].nc = 1
        print(f"Reconfigured model head for {model.model.model[-1].nc} class.")
    else:
        print("Could not reconfigure model head. Proceeding with original configuration.")

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        lr0=learning_rate,
        name=f"yolo11n_obb_hp_set_{i+1}" # Unique name for each run
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()  # evaluate model performance on the validation set

    # Export the trained model to a format suitable for deployment
    # Save with a unique name based on the hyperparameter set.
    # These ONNX files will be saved in the current working directory.
    # Training results (weights, plots, etc.) will be saved in the 'runs/' directory
    # under uniquely named folders (e.g., runs/obb/yolo11n_obb_hp_set_1).
    model.export(format="onnx", name=f"yolo11n-obb_hp_set_{i+1}")
