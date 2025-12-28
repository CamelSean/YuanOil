import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import os
import yaml
import argparse
from pathlib import Path

def calculate_dataset_stats(dataset_path, save_yaml=True):
    """
    計算資料集的 Mean 和 Std，並儲存為 dataset_stats.yaml
    """
    dataset_path = Path(dataset_path)
    # 假設訓練圖片在 images/train 或 images
    image_dir = dataset_path / 'images' / 'train'
    if not image_dir.exists():
        image_dir = dataset_path / 'images'
        if not image_dir.exists():
            print(f"[Error] Cannot find images directory in {dataset_path}")
            return

    print(f"Scanning images in: {image_dir}")
    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    
    if not image_files:
        print("[Error] No images found.")
        return

    # 初始化累加變數
    # 為了通用性，我們先讀取第一張圖片來確定通道數
    first_img = np.array(Image.open(image_files[0]))
    if first_img.ndim == 2:
        channels = 1
    else:
        channels = first_img.shape[2]
    
    print(f"Detected {channels} channels.")
    
    channel_sum = np.zeros(channels)
    channel_sum_sq = np.zeros(channels)
    pixel_count = 0

    print("Calculating Mean and Std...")
    for img_path in tqdm(image_files):
        try:
            img = Image.open(img_path)
            if channels == 1:
                img = img.convert('L')
            else:
                img = img.convert('RGB') # 假設多通道是 RGB
                
            img_np = np.array(img) / 255.0 # Normalize to [0, 1]
            
            if channels == 1:
                # (H, W) -> (H, W, 1) for consistent processing
                img_np = np.expand_dims(img_np, axis=-1)
            
            # 確保維度正確
            if img_np.shape[2] != channels:
                print(f"[Warning] Skipping {img_path}: Expected {channels} channels, got {img_np.shape[2]}")
                continue

            h, w, c = img_np.shape
            pixel_count += h * w
            
            # Sum over height and width
            channel_sum += np.sum(img_np, axis=(0, 1))
            channel_sum_sq += np.sum(np.square(img_np), axis=(0, 1))
            
        except Exception as e:
            print(f"[Warning] Error processing {img_path}: {e}")

    if pixel_count == 0:
        print("[Error] No valid pixels processed.")
        return

    # Calculate Mean and Std
    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sum_sq / pixel_count - np.square(mean))

    print("\n" + "="*40)
    print("Results:")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print("="*40)

    if save_yaml:
        stats = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        output_path = dataset_path / 'dataset_stats.yaml'
        with open(output_path, 'w') as f:
            yaml.dump(stats, f)
        print(f"\nStats saved to: {output_path}")
        print("You can now use this dataset for training, and the adapters will automatically load these values.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dataset Mean and Std")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset root (containing images/train)")
    args = parser.parse_args()
    
    calculate_dataset_stats(args.dataset_path)
