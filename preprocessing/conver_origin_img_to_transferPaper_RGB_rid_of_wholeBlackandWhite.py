import os
import cv2
import numpy as np
import shutil
import gc
from tqdm import tqdm
from pathlib import Path

# ================= 設定區域 =================
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

INPUT_ROOT = r"/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/00_Raw_Source"
OUTPUT_ROOT = r"/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/01_Feature_Engineering/Pseudo_RGB_Full"

TILE_SIZE = 2048

# 設定要忽略的背景值
# 如果您的無效背景只有純黑，設為 [0]
# 如果有純黑也有純白，設為 [0, 255]
IGNORE_VALUES = [0, 255] 
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_valid_mask(img):
    """
    建立有效像素遮罩，排除 IGNORE_VALUES 中定義的背景值
    """
    # 預設全部有效 (255)
    mask = np.full(img.shape, 255, dtype=np.uint8)
    
    for val in IGNORE_VALUES:
        # 將無效值的位置設為 0
        mask[img == val] = 0
        
    return mask

def get_global_stats_and_background(img):
    """
    計算抗干擾的全域統計數據 (排除背景值)
    """
    # 1. 建立遮罩
    mask = get_valid_mask(img)
    
    # 檢查是否整張圖都是無效值 (避免 crash)
    valid_pixel_count = cv2.countNonZero(mask)
    if valid_pixel_count == 0:
        # 假如全黑，返回預設值，避免除以零
        return 0, 1, 0, 1, np.zeros_like(img, dtype=np.float32), np.zeros_like(img, dtype=np.float32)

    # 2. Masked Mean/Std (只計算有效區域)
    mean_val, std_val = cv2.meanStdDev(img, mask=mask)
    mean_vv = mean_val[0][0]
    std_vv = std_val[0][0]

    # 3. Masked Min/Max (只計算有效區域)
    min_val, max_val, _, _ = cv2.minMaxLoc(img, mask=mask)
    
    # 4. B Channel 背景表面處理 (Inpainting 策略)
    # 論文建議：用平均值填充無效區域，以避免模糊時產生邊緣誤差 
    
    h, w = img.shape
    stride = 32
    small_h, small_w = h // stride, w // stride
    if small_h == 0: small_h = 1
    if small_w == 0: small_w = 1

    # 縮小圖像
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    # [關鍵步驟] 將小圖中的無效區域 (mask==0) 替換為全域平均值 (mean_vv)
    # 這樣 GaussianBlur 就不會被 0 拉低數值
    img_small[mask_small == 0] = mean_vv

    # 計算局部標準差
    img_sq_small = cv2.resize(img.astype(np.float32)**2, (small_w, small_h), interpolation=cv2.INTER_AREA)
    # 同樣填充平方圖的無效區域
    img_sq_small[mask_small == 0] = (mean_vv ** 2) + (std_vv ** 2) # E[X^2] approximation

    diff = img_sq_small - (img_small ** 2)
    diff[diff < 0] = 0
    std_small = np.sqrt(diff)

    # Gaussian Blur
    background_mean = cv2.GaussianBlur(img_small, (7, 7), 0)
    background_std = cv2.GaussianBlur(std_small, (7, 7), 0)

    return mean_vv, std_vv, min_val, max_val, background_mean, background_std

def process_tile(vv_tile, global_mean, global_std, global_min, global_max, bg_mean_tile, bg_std_tile):
    """
    處理單一圖塊，並保留原本的無效區域為 0 (黑色)
    """
    epsilon = 1e-6
    vv = vv_tile.astype(np.float32)
    
    # 建立該 Tile 的遮罩，用於最後還原背景
    tile_mask = get_valid_mask(vv_tile)

    # --- 1. R Channel ---
    # r = np.log10(vv + epsilon)
    # g_r_min = np.log10(global_min + epsilon)
    # g_r_max = np.log10(global_max + epsilon)
    
    # if g_r_max - g_r_min > 0:
    #     r_norm = (r - g_r_min) / (g_r_max - g_r_min)
    # else:
    #     r_norm = np.zeros_like(r)
    
    # R = (np.clip(r_norm, 0, 1) * 255).astype(np.uint8)


    # 如果輸入已經是 0-255 且適合觀看，不要再取 log10
    # 直接做 Min-Max 確保對比度拉滿即可
    # 這裡我們假設 vv_tile 已經是 0-255 的數值
    # 方案 A: 如果原圖對比度已經很好，直接用
    # R = vv_tile.astype(np.uint8) 
    
    # 方案 B (推薦): 為了保險，還是做一次 Min-Max Scaling (但不做 Log)
    # 這是為了確保 R 通道能佔滿 0-255 的動態範圍，符合論文 "scale to [0, 255]" 的精神
    r_float = vv_tile.astype(np.float32)
    # 使用傳入的全域 Min/Max (注意：傳入的 global_min/max 也要是基於 0-255 的)
    if global_max - global_min > 0:
        r_norm = (r_float - global_min) / (global_max - global_min)
    else:
        r_norm = np.zeros_like(r_float)
    
    R = (np.clip(r_norm, 0, 1) * 255).astype(np.uint8)

    # --- 2. G Channel ---
    g_raw = np.arctan((vv - global_mean) / (global_std + epsilon))
    g_norm = (g_raw / np.pi) + 0.5 
    G = (np.clip(g_norm, 0, 1) * 255).astype(np.uint8)

    # --- 3. B Channel ---
    b_star = (vv - bg_mean_tile) / (bg_std_tile + epsilon)
    b_norm = (np.arctan(b_star) / np.pi) + 0.5
    B = (np.clip(b_norm, 0, 1) * 255).astype(np.uint8)

    # --- 4. Post-processing: 還原背景 ---
    # 如果原圖是 0 (背景)，轉換後應該保持為 0 (黑色)，而不是被數學公式算出一個灰色值
    # 這一步對於可視化和訓練都很重要
    
    merged = cv2.merge([B, G, R])
    # 使用 mask 將背景強制設回 0
    merged = cv2.bitwise_and(merged, merged, mask=tile_mask)
    
    return merged

def sar_to_rgb_tiled(image_path):
    try:
        full_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Read Error: {e}")
        return None
    
    if full_img is None: return None

    if full_img.ndim == 3:
        full_img = full_img[:, :, 0]
    
    h_full, w_full = full_img.shape
    
    # 1. 計算全域參數 (Masked Statistics)
    g_mean, g_std, g_min, g_max, bg_mean_small, bg_std_small = get_global_stats_and_background(full_img)
    small_h, small_w = bg_mean_small.shape
    
    # 如果整張圖都是空的 (極端情況)
    if g_max == 0 and g_min == 0:
         return np.zeros((h_full, w_full, 3), dtype=np.uint8)

    # 2. 準備輸出
    try:
        result_img = np.zeros((h_full, w_full, 3), dtype=np.uint8)
    except MemoryError:
        print(f"記憶體不足 ({h_full}x{w_full})，跳過。")
        return None

    # 3. 分塊迴圈
    for y in range(0, h_full, TILE_SIZE):
        for x in range(0, w_full, TILE_SIZE):
            y_end = min(y + TILE_SIZE, h_full)
            x_end = min(x + TILE_SIZE, w_full)
            curr_h = y_end - y
            curr_w = x_end - x
            
            patch_vv = full_img[y:y_end, x:x_end]
            
            # 背景 Remap
            x_indices = np.arange(x, x + curr_w, dtype=np.float32) * (small_w / w_full)
            y_indices = np.arange(y, y + curr_h, dtype=np.float32) * (small_h / h_full)
            mesh_x, mesh_y = np.meshgrid(x_indices, y_indices)
            
            bg_mean_final = cv2.remap(bg_mean_small, mesh_x, mesh_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            bg_std_final = cv2.remap(bg_std_small, mesh_x, mesh_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            # 核心處理
            processed_patch = process_tile(patch_vv, g_mean, g_std, g_min, g_max, bg_mean_final, bg_std_final)
            
            result_img[y:y_end, x:x_end] = processed_patch
            
            del patch_vv, processed_patch, bg_mean_final, bg_std_final, mesh_x, mesh_y
    
    del full_img, g_mean, g_std, g_min, g_max, bg_mean_small, bg_std_small
    gc.collect()
    
    return result_img

def process_dataset():
    input_path = Path(INPUT_ROOT)
    output_path = Path(OUTPUT_ROOT)

    print(f"開始處理資料集 (Masked v5)...")
    print(f"忽略背景值: {IGNORE_VALUES}")
    
    file_list = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    for src_file_path in tqdm(file_list, desc="Processing files"):
        src_path_obj = Path(src_file_path)
        relative_path = src_path_obj.relative_to(input_path)
        dest_file_path = output_path / relative_path
        ensure_dir(dest_file_path.parent)

        if "labels" in src_path_obj.parts:
            shutil.copy2(src_file_path, dest_file_path)
        elif "images" in src_path_obj.parts:
            if src_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                try:
                    processed_img = sar_to_rgb_tiled(str(src_file_path))
                    if processed_img is not None:
                        cv2.imwrite(str(dest_file_path), processed_img)
                        del processed_img
                        gc.collect()
                except Exception as e:
                    print(f"Error: {src_file_path}: {e}")
                    gc.collect()
            else:
                shutil.copy2(src_file_path, dest_file_path)
        else:
            shutil.copy2(src_file_path, dest_file_path)

if __name__ == "__main__":
    process_dataset()