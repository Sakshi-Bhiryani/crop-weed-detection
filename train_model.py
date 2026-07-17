import os
import shutil
from ultralytics import YOLO

def main():
    # ==============================================================================
    # 1. INITIALIZE TRAINING FRAMEWORK
    # ==============================================================================
    print(" Initializing YOLOv8 Object Detection Pipeline...")
    
    # Load the standard foundational Nano baseline model weights
    model = YOLO("yolov8n.pt")
    
    # Specify project paths
    yaml_config = "data.yaml"
    
    # Verify that the configuration blueprint layout file is present
    if not os.path.exists(yaml_config):
        raise FileNotFoundError(f" Error: Config file '{yaml_config}' was not found in the root directory!")

    # ==============================================================================
    # 2. RUN MODEL TRAINING CYCLE
    # ==============================================================================
    print("\n🌱 Starting Crop and Weed Custom Feature Learning...")
    
    # Execute the training pipeline optimization routine
    results = model.train(
        data=yaml_config,   # Target data configuration paths blueprint
        epochs=3,           # Number of complete training cycles
        imgsz=512,          # Target image resolution dimensions 
        batch=16,           # Training batch input volume capacity 
        workers=2,          # Parallel data pipeline extraction worker subprocesses
        device="cpu"        # Standard processing engine target selection
    )
    
    print("\n Model Training Phase Completed successfully!")

    # ==============================================================================
    # 3. EXPORT TARGET MODEL WEIGHTS FOR DEPLOYMENT
    # ==============================================================================
    # Extract the exact folder location where YOLO saved the final iteration runs
    save_dir = results.save_dir
    source_weights_path = os.path.join(save_dir, "weights", "best.pt")
    destination_weights_path = "best.pt"
    
    # Safely duplicate model weights back to project root workspace folder
    if os.path.exists(source_weights_path):
        shutil.copy(source_weights_path, destination_weights_path)
        print(f" Custom weights successfully linked to app workspace: '{destination_weights_path}'")
    else:
        print(f" Warning: Model weights could not be automatically located at {source_weights_path}")

if __name__ == "__main__":
    main()