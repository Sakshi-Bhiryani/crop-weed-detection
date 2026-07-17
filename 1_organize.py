import os
import shutil
import random

def split_flat_dataset(source_dir="agri_data", output_dir="dataset", split_ratio=(0.7, 0.2, 0.1)):

    # 1. Create the structured destination directories for YOLO 
    splits = ['train', 'val', 'test']
    subfolders = ['images', 'labels']

    for split in splits:
        for subfolder in subfolders:
            os.makedirs(os.path.join(output_dir, split, subfolder), exist_ok=True)
    
    if not os.path.exists(source_dir):
        print(f"Error: The directory '{source_dir}' does not exist!")
        return

    # 2. Gather all image files directly from the flat source directory 
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPEG')
    all_files = os.listdir(source_dir)

    # Extract unique base filenames (e.g., 'agri_0_3') from the image files
    base_names = [
        os.path.splitext(f)[0]
        for f in all_files 
        if f.lower().endswith(image_extensions)
    ]

    if not base_names:
        print(f"No images found directly inside the '{source_dir}' folder!")
        return
    
    # Shuffle the filenames randomly to mix crops and weeds evenly
    random.seed(42)
    random.shuffle(base_names)

    # 3. Calculate splitting boundaries
    total_images = len(base_names)
    train_bound = int(total_images * split_ratio[0])
    val_bound = train_bound + int(total_images * split_ratio[1])

    # 4. Copy files over to their proper structured folders 
    for index, name in enumerate(base_names):
        # Determine target split folder
        if index < train_bound:
            target_split = 'train'
        elif index < val_bound:
            target_split = 'val'
        else:
            target_split = 'test'

        # Find the correct original image name with extension 
        img_file = next((f for f in all_files if os.path.splitext(f)[0] == name and f.lower().endswith(image_extensions)), None)
        # Find the corresponding label file
        label_file = f"{name}.txt"

        # Safe copy execution
        if img_file:
            shutil.copy(os.path.join(source_dir, img_file), os.path.join(output_dir, target_split, 'images', img_file))

        if label_file in all_files:
            shutil.copy(os.path.join(source_dir, label_file), os.path.join(output_dir, target_split, 'labels', label_file))

    print(f"Successfully sorted {total_images} items into target training splits!") 

if __name__ == "__main__":
        split_flat_dataset()

