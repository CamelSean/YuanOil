
import argparse
import os
from pathlib import Path
import shutil
from PIL import Image
from collections import Counter
import cv2
import numpy as np
from tqdm import tqdm

def analyze_and_prepare_dataset(input_dir: Path, output_dir: Path, target_size: tuple[int, int]):
    """
    Analyzes, copies, and preprocesses a dataset for training.

    - Deletes and recreates the output directory.
    - Analyzes image sizes and mask formats from the input directory.
    - Resizes images and segmentation masks to the target size.
    - Converts masks to 8-bit single-channel format.
    - Copies YOLO .txt labels without modification.
    - Saves processed files to the output directory.

    Args:
        input_dir (Path): Path to the source dataset.
        output_dir (Path): Path to the destination for the processed dataset.
        target_size (tuple[int, int]): The target (width, height) for resizing.
    """
    # --- 1. Handle Output Directory ---
    if output_dir.exists():
        print(f"Output directory '{output_dir}' already exists. Deleting it.")
        shutil.rmtree(output_dir)
    print(f"Creating new output directory: '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Analyze the source dataset ---
    print("\n--- Analyzing Source Dataset ---")
    image_paths = list(input_dir.glob("images/**/*.png"))
    label_mask_paths = list(input_dir.glob("labels/**/*.png"))

    if not image_paths:
        print("No images found in the input directory. Exiting.")
        return

    # Analyze image dimensions
    image_dims_counter = Counter()
    for p in tqdm(image_paths, desc="Analyzing image dimensions"):
        try:
            with Image.open(p) as img:
                image_dims_counter[img.size] += 1
        except Exception as e:
            print(f"Warning: Could not read image {p}: {e}")

    print("\nImage Size Distribution:")
    for size, count in image_dims_counter.items():
        print(f"- {size[0]}x{size[1]}: {count} images")

    # Analyze mask properties
    mask_info_counter = Counter()
    for p in tqdm(label_mask_paths, desc="Analyzing label masks"):
        try:
            with Image.open(p) as img:
                mask_info_counter[f"Mode: {img.mode}, Size: {img.size}"] += 1
        except Exception as e:
            print(f"Warning: Could not read mask {p}: {e}")
    
    print("\nLabel Mask (PNG) Distribution:")
    if not mask_info_counter:
        print("- No PNG masks found.")
    for info, count in mask_info_counter.items():
        print(f"- {info}: {count} masks")
    print("--- Analysis Complete ---\n")


    # --- 3. Process and Copy Files ---
    print(f"--- Processing and Resizing to {target_size[0]}x{target_size[1]} ---")
    all_files = list(input_dir.glob("**/*"))
    
    for src_path in tqdm(all_files, desc="Processing files"):
        if src_path.is_dir():
            continue

        relative_path = src_path.relative_to(input_dir)
        dst_path = output_dir / relative_path

        # Create parent directory in destination if it doesn't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = src_path.suffix.lower()

        try:
            if suffix == '.png':
                is_label = 'labels' in src_path.parts
                
                img = Image.open(src_path)
                
                # For labels, we must use NEAREST resampling to avoid creating new pixel values.
                resample_method = Image.Resampling.NEAREST if is_label else Image.Resampling.LANCZOS
                
                img_resized = img.resize(target_size, resample=resample_method)

                if is_label:
                    # Ensure the mask is saved in a human-readable RGB format.
                    # The training pipeline will handle the conversion to a single-channel tensor.
                    if img_resized.mode != 'RGB':
                        img_resized = img_resized.convert('RGB')
                
                img_resized.save(dst_path)

            elif suffix == '.txt':
                # YOLO .txt files contain relative coordinates.
                # The training pipeline handles resizing, so we just copy the files.
                shutil.copy2(src_path, dst_path)
            
            else:
                # Copy any other files (e.g., .jpg, .json)
                shutil.copy2(src_path, dst_path)

        except Exception as e:
            print(f"\nError processing file {src_path}: {e}")
            print("Skipping this file.")

    print("\n--- Dataset Preparation Complete ---")
    print(f"Processed dataset saved to: {output_dir}")


if __name__ == "__main__":
    # --- 在這裡設定預設路徑 ---
    # 如果不想每次都從命令列輸入，可以直接在這裡修改
    DEFAULT_INPUT_DIR = '/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/02_Patched_HighRes/VV_1Ch_Patch_2048_Separated' # "/path/to/your/original_dataset"
    DEFAULT_OUTPUT_DIR = '/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/03_Training_Ready_512/VV_1Ch_2048_Resize_512_Separated' # "/path/to/your/processed_dataset"
    # -------------------------

    parser = argparse.ArgumentParser(
        description="Analyze and prepare a dataset for deep learning by resizing images and masks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Path to the source dataset directory.\n"
             "Expected structure:\n"
             "- input_dir/\n"
             "  - images/\n"
             "    - train/\n"
             "    - val/\n"
             "    - test/\n"
             "  - labels/\n"
             "    - train/\n"
             "    - val/\n"
             "    - test/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to the output directory where the processed dataset will be saved."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Target image size as two integers: width height (default: 512 512)."
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    target_image_size = tuple(args.imgsz)

    if not input_path.is_dir() or "your/original_dataset" in str(input_path):
        print(f"Error: Input directory not found or default path is not changed.")
        print(f"Please edit the script to set 'DEFAULT_INPUT_DIR' or provide '--input-dir' via command line.")
        print(f"Current input path: '{input_path}'")
    else:
        analyze_and_prepare_dataset(input_path, output_path, target_image_size)

    # Example usage from the command line:
    # python prepare_dataset.py --input-dir /path/to/original --output-dir /path/to/processed --imgsz 640 640
    # python /home/yuan/Oil_Project_10-8/preprocessing/prepare_dataset.py --input-dir /home/yuan/Oil_Project_10-8/dataset/datasetv4/DV4_SAR_Big_Patch/Patched_P2048_O512_BG100p/DV4_SAR_Big --output-dir /home/yuan/Oil_Project_10-8/dataset/datasetv4/DV4_SAR_Big_Patch/Patched_P2048_O512_BG100p/DV4_SAR_Big_Resize512 --imgsz 512 512
    # 