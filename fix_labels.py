import os

def fix_label_file(filepath):
    """Reads a label file, corrects class IDs to 0, normalizes coordinates, and writes back."""
    corrected_lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        try:
            # OBB format: class_id x_center y_center width height angle
            # For OBB, there are 6 or 9 values (if rotated bounding box with 4 points)
            # Assuming 6 values: class_id, x, y, w, h, angle
            # Or 9 values: class_id, x1, y1, x2, y2, x3, y3, x4, y4
            
            # The error message "Label class X exceeds dataset class count 1" implies the first element is class_id.
            # The "non-normalized or out of bounds coordinates" implies subsequent elements are coordinates.

            class_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]

            # Correct class ID to 0
            corrected_class_id = 0

            # Normalize coordinates to be within [0, 1]
            # Assuming coordinates are x_center, y_center, width, height, angle
            # Or x1, y1, x2, y2, x3, y3, x4, y4
            # For simplicity, I'll clamp all float coordinates to [0, 1]
            # Angle is typically in degrees or radians, and might not be clamped to [0,1]
            # For OBB, angle is usually in radians [-pi/2, pi/2] or degrees [-90, 90]
            # I will only clamp x,y,w,h. The angle should be left as is.
            
            # If it's 6 parts (class_id, x, y, w, h, angle)
            if len(parts) == 6:
                # Clamp x, y, w, h
                coords[0] = max(0.0, min(1.0, coords[0])) # x_center
                coords[1] = max(0.0, min(1.0, coords[1])) # y_center
                coords[2] = max(0.0, min(1.0, coords[2])) # width
                coords[3] = max(0.0, min(1.0, coords[3])) # height
                # Angle (coords[4]) is left as is
                corrected_line = f"{corrected_class_id} {' '.join(map(str, coords))}\n"
            # If it's 9 parts (class_id, x1, y1, x2, y2, x3, y3, x4, y4)
            elif len(parts) == 9:
                # Clamp all 8 coordinates
                for j in range(8):
                    coords[j] = max(0.0, min(1.0, coords[j]))
                corrected_line = f"{corrected_class_id} {' '.join(map(str, coords))}\n"
            else:
                # If format is unexpected, keep original line but correct class_id
                print(f"Warning: Unexpected number of parts ({len(parts)}) in {filepath}: {line.strip()}. Only class ID corrected.")
                parts[0] = str(corrected_class_id)
                corrected_line = ' '.join(parts) + '\n'

            corrected_lines.append(corrected_line)

        except ValueError as e:
            print(f"Error parsing line in {filepath}: {line.strip()} - {e}. Skipping line.")
            continue

    with open(filepath, 'w') as f:
        f.writelines(corrected_lines)
    print(f"Fixed: {filepath}")

def process_directory(base_dir):
    """Processes all label files in train/labels and valid/labels subdirectories."""
    label_dirs = [
        os.path.join(base_dir, 'train', 'labels'),
        os.path.join(base_dir, 'valid', 'labels'),
        os.path.join(base_dir, 'test', 'labels') # Include test labels if they exist
    ]

    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            print(f"Label directory not found: {label_dir}. Skipping.")
            continue

        print(f"Processing labels in: {label_dir}")
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(label_dir, filename)
                fix_label_file(filepath)

if __name__ == "__main__":
    # Base directory of the dataset, derived from data/book_only.yaml
    dataset_base_dir = "data/LibVision 2 -No Other Class-.v1i.yolov8-obb"
    
    if not os.path.exists(dataset_base_dir):
        print(f"Error: Dataset base directory not found at {dataset_base_dir}")
    else:
        process_directory(dataset_base_dir)
    print("Label fixing process complete.")
