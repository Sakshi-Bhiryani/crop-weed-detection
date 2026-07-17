import os
import shutil
import glob
from datetime import datetime

# ==============================================================================
# 1. DIRECTORY & WORKSPACE MANAGEMENT
# ==============================================================================
def ensure_dataset_structure(base_path="dataset"):
    """
    Creates the required YOLO train/val/test split directory architecture
    if it doesn't already exist in the workspace.
    """
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'labels']
    
    created_count = 0
    for split in splits:
        for subdir in subdirs:
            target_dir = os.path.join(base_path, split, subdir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                created_count += 1
                
    if created_count > 0:
        print(f" Created {created_count} missing split subdirectories inside '{base_path}/'.")
    else:
        print(f"✓ Dataset directory structure verified and ready.")

# ==============================================================================
# 2. DATASET AUDITING & VALIDATION
# ==============================================================================
def verify_dataset_integrity(base_path="dataset"):
    """
    Scans the train, val, and test splits to ensure that every image file 
    has a corresponding YOLO label text file.
    """
    print(f"\n Auditing dataset integrity inside '{base_path}'...")
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = os.path.join(base_path, split, 'images')
        lbl_dir = os.path.join(base_path, split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f" Skipping {split} split - directories missing.")
            continue
            
        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')}
        
        # Mismatches tracking
        missing_labels = images - labels
        missing_images = labels - images
        
        print(f" Split [{split.upper()}]: Found {len(images)} images and {len(labels)} labels.")
        
        if missing_labels:
            print(f"    Alert: {len(missing_labels)} images lack a matching .txt label file!")
        if missing_images:
            print(f"    Alert: {len(missing_images)} label files lack a matching source image!")
            
        if not missing_labels and not missing_images:
            print(f"   ✓ 100% Match: Image and label pairs are perfectly aligned.")

# ==============================================================================
# 3. METADATA REPORT GENERATION
# ==============================================================================
def generate_project_summary(project_dir="."):
    """
    Generates a quick log file summarizing the state of your project weights,
    dataset status, and the latest execution timestamp.
    """
    summary_file = os.path.join(project_dir, "project_summary.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check for core project components
    has_weights = os.path.exists(os.path.join(project_dir, "best.pt"))
    has_yaml = os.path.exists(os.path.join(project_dir, "data.yaml"))
    
    # Count dataset components safely
    total_images = len(glob.glob(os.path.join(project_dir, "dataset", "**", "images", "*.*"), recursive=True))
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("🌱 CROP & WEED DETECTION PROJECT SYSTEM SUMMARY  \n")
        f.write("==================================================\n")
        f.write(f"Generated On      : {timestamp}\n")
        f.write(f"Total Images split: {total_images} items\n")
        f.write(f"Data Blueprint    : {'✓ Valid (data.yaml detected)' if has_yaml else '✗ Missing (data.yaml)'}\n")
        f.write(f"Trained Core Model: {'✓ Ready (best.pt detected)' if has_weights else '⏳ Pending Training'}\n")
        f.write("==================================================\n")
        
    print(f"\n Project status report successfully written to '{summary_file}'")

# Simple operational test wrapper block
if __name__ == "__main__":
    print(" Running core utility system diagnostic tests...")
    ensure_dataset_structure()
    verify_dataset_integrity()
    generate_project_summary()