import os
import sys
import glob
import random
import shutil
import zipfile
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Extract and automatically organize IAM and READ2016 datasets into correct HTR-ConvText format.")
    parser.add_argument("--iam_zip", type=str, required=True, help="Path to IAM_lines.zip")
    parser.add_argument("--read_zip", type=str, required=True, help="Path to read2016_lines.zip")
    return parser.parse_args()

def prepare_dataset(zip_path, dataset_name, ext):
    print(f"\\n--- Preparing {dataset_name} dataset ---")
    data_dir = os.path.join(os.path.dirname(__file__), "data", dataset_name.lower())
    lines_dir = os.path.join(data_dir, "lines")
    
    # Cleanup previous instances
    if os.path.exists(data_dir):
        print(f"Cleaning up previous {data_dir}...")
        shutil.rmtree(data_dir)
        
    os.makedirs(lines_dir, exist_ok=True)
    temp_dir = os.path.join(data_dir, "temp_unzip")
    
    # Step 1: Extract files
    print(f"Extracting {zip_path} into a temporary folder... (This may take a minute)")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Step 2: Grab all images and matching txt
    print(f"Finding all {ext} files and matching .txt transcriptions...")
    images_found = glob.glob(os.path.join(temp_dir, f"**/*{ext}"), recursive=True)
    
    valid_pairs = []
    
    for img_path in images_found:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        dir_name = os.path.dirname(img_path)
        txt_path = os.path.join(dir_name, base_name + ".txt")
        
        if os.path.exists(txt_path):
            # Move flatly to the lines directory
            dest_img = os.path.join(lines_dir, os.path.basename(img_path))
            dest_txt = os.path.join(lines_dir, os.path.basename(txt_path))
            
            shutil.move(img_path, dest_img)
            shutil.move(txt_path, dest_txt)
            
            valid_pairs.append(os.path.basename(dest_img))
    
    # Safely remove temporary unzip directory
    shutil.rmtree(temp_dir)
    print(f"Extracted and verified {len(valid_pairs)} valid image-text pairs.")
    
    if len(valid_pairs) == 0:
        print(f"Error: Could not find any valid pairs for {dataset_name}! Exiting.")
        sys.exit(1)
        
    # Step 3: Split and generate .ln files (80-10-10 split)
    print("Generating train.ln, val.ln, and test.ln with 80-10-10 split...")
    random.shuffle(valid_pairs)
    
    total = len(valid_pairs)
    train_idx = int(0.8 * total)
    val_idx = train_idx + int(0.1 * total)
    
    train_set = valid_pairs[:train_idx]
    val_set = valid_pairs[train_idx:val_idx]
    test_set = valid_pairs[val_idx:]
    
    with open(os.path.join(data_dir, "train.ln"), "w") as f:
        f.write("\n".join(train_set))
    with open(os.path.join(data_dir, "val.ln"), "w") as f:
        f.write("\n".join(val_set))
    with open(os.path.join(data_dir, "test.ln"), "w") as f:
        f.write("\n".join(test_set))
        
    print(f"Done! Created: \n -> train.ln ({len(train_set)}) \n -> val.ln ({len(val_set)}) \n -> test.ln ({len(test_set)})")

if __name__ == "__main__":
    args = get_args()
    
    # Random seed for repeatable splits
    random.seed(42)
    
    prepare_dataset(args.iam_zip, "iam", ".png")
    prepare_dataset(args.read_zip, "read2016", ".jpeg")
    
    print("\\n🚀 All datasets successfully set up! You are now ready to train.")
