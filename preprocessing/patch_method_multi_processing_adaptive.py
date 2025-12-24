# ===================================================================
# ===      preprocessing/patch_method.py (v1.5 - 雙重疊設定版)      ===
# ===  (v1.4 雙軌策略 + v1.5 獨立設定 Train/Val 與 Test 重疊量)      ===
# ===================================================================
# python create_patched_dataset_with_log.py
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import pandas as pd
from datetime import datetime
import concurrent.futures 

# PIL 可能會對大圖有警告，設定此項可避免
Image.MAX_IMAGE_PIXELS = None

# --- CPU 核心數設定 ---
# None 表示使用所有可用的核心
NUM_WORKERS = None 

# --- 來源與輸出設定 ---
SOURCE_BASE_DIR = "/home/sean/oil_11_26/dataset"
OUTPUT_BASE_DIR = "/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/DV4_SAR_Small_v3_relabel_2048_Patch"
CATEGORIES = ["DV4_SAR_Small_v3_relabel"]
SPLITS = ["train", "val", "test"]

# --- Patching 參數設定 ---
PATCH_SIZE = 2048

# [v1.5 新增] 分別設定 Train/Val 和 Test 的重疊量
# 1. Train/Val: 僅針對"正樣本"使用此重疊量 (增加多樣性)，背景不重疊。
OVERLAP_TRAIN_VAL = 512

# 2. Test: 針對"整張圖"使用此重疊量 (標準滑動窗口)。
#    建議設大一點 (例如 256 或 512) 以獲得更好的重建 (Reconstruction) 效果，減少接縫感。
OVERLAP_TEST = 512       

RANDOM_SEED = 42

# --- 背景樣本保留比例設定 ---
# (僅作用於 Train/Val 的背景樣本)
BACKGROUND_KEEP_RATIO = 1

# --- 功能開關 ---
SEPARATE_OUTPUT = False 
 
# --- PNG 轉 TXT 參數設定 ---
TXT_GENERATION_PARAMS = {
    "class_id": 0,
    "target_pixel_value": 255,
    "epsilon_factor": 0.002,
    "min_contour_area": 1.0
}

# ===================================================================
# === 核心函式
# ===================================================================

def check_target_pixels_exist(mask_segment_pil, target_value):
    """檢查一個遮罩圖塊中，是否存在指定的目標像素值。"""
    segment_array = np.array(mask_segment_pil)
    return np.any(segment_array == target_value)

def slice_image_standard_generator(image_pil, mask_pil, patch_size, overlap, target_pixel_val):
    """
    [v1.4] 標準均勻生成器 (給 Test 使用)
    策略：無論內容為何，統一使用固定重疊量的滑動窗口。
    """
    image_width, image_height = image_pil.size
    pad_color_image = (255, 255, 255) if image_pil.mode == 'RGB' else 255
    pad_value_mask = 0 

    stride = patch_size - overlap
    
    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            # 1. 裁切
            crop_box = (x, y, min(x + patch_size, image_width), min(y + patch_size, image_height))
            patch_img = image_pil.crop(crop_box)
            patch_mask = mask_pil.crop(crop_box)
            
            # 2. 填充 (統一左上角貼上 [v1.1 Fix])
            padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
            padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
            padded_img.paste(patch_img, (0, 0))
            padded_seg.paste(patch_mask, (0, 0))
            
            # 3. 判斷是否包含目標
            is_positive = check_target_pixels_exist(patch_mask, target_pixel_val)
            
            # 回傳
            yield padded_img, padded_seg, x, y, is_positive


def slice_image_adaptive_generator(image_pil, mask_pil, patch_size, positive_overlap, target_pixel_val):
    """
    [v1.3] 自適應步長生成器 (給 Train/Val 使用)
    策略：油汙密集 (Overlap)，背景稀疏 (No Overlap)。
    """
    image_width, image_height = image_pil.size
    pad_color_image = (255, 255, 255) if image_pil.mode == 'RGB' else 255
    pad_value_mask = 0 

    # --- Pass 1: 密集掃描 (只抓正樣本) ---
    stride_pos = patch_size - positive_overlap
    for y in range(0, image_height, stride_pos):
        for x in range(0, image_width, stride_pos):
            crop_box = (x, y, min(x + patch_size, image_width), min(y + patch_size, image_height))
            patch_mask = mask_pil.crop(crop_box)
            
            if check_target_pixels_exist(patch_mask, target_pixel_val):
                patch_img = image_pil.crop(crop_box)
                padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
                padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
                padded_img.paste(patch_img, (0, 0))
                padded_seg.paste(patch_mask, (0, 0))
                yield padded_img, padded_seg, x, y, True

    # --- Pass 2: 稀疏掃描 (只抓負樣本) ---
    stride_neg = patch_size  # 不重疊
    for y in range(0, image_height, stride_neg):
        for x in range(0, image_width, stride_neg):
            crop_box = (x, y, min(x + patch_size, image_width), min(y + patch_size, image_height))
            patch_mask = mask_pil.crop(crop_box)
            
            if not check_target_pixels_exist(patch_mask, target_pixel_val):
                patch_img = image_pil.crop(crop_box)
                padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
                padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
                padded_img.paste(patch_img, (0, 0))
                padded_seg.paste(patch_mask, (0, 0))
                yield padded_img, padded_seg, x, y, False


def convert_mask_png_to_yolo_txt(png_path, output_txt_path, params):
    try:
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: return "error"
        height, width = mask.shape
        
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == params["target_pixel_value"]] = 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_format_lines = []
        for contour in contours:
            if cv2.contourArea(contour) < params["min_contour_area"]: continue
            epsilon = params["epsilon_factor"] * cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_polygon) >= 3:
                normalized_points = []
                for point_wrapper in approx_polygon:
                    point = point_wrapper[0]
                    norm_x = max(0.0, min(1.0, point[0] / width))
                    norm_y = max(0.0, min(1.0, point[1] / height))
                    normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                if normalized_points:
                    yolo_format_lines.append(f"{params['class_id']} {' '.join(normalized_points)}")
        
        with open(output_txt_path, 'w') as f:
            if yolo_format_lines:
                f.write("\n".join(yolo_format_lines) + "\n")
                return "contours_written"
            else:
                return "empty_file_written"
    except Exception as e:
        print(f"  [ERROR] 處理檔案 '{os.path.basename(png_path)}' 時發生錯誤: {e}")
        return "error"

def process_single_image(job_args):
    (
        image_filename, source_img_dir, source_mask_dir, main_output_path,
        category, split, patch_size, overlap, effective_bg_keep_ratio,
        separate_output, txt_generation_params
    ) = job_args

    # 建立輸出目錄
    if separate_output:
        output_img_dir_pos = os.path.join(main_output_path, category, "images", f"{split}_pos")
        output_label_dir_pos = os.path.join(main_output_path, category, "labels", f"{split}_pos")
        output_img_dir_neg = os.path.join(main_output_path, category, "images", f"{split}_neg")
        output_label_dir_neg = os.path.join(main_output_path, category, "labels", f"{split}_neg")
        for d in [output_img_dir_pos, output_label_dir_pos, output_img_dir_neg, output_label_dir_neg]:
            os.makedirs(d, exist_ok=True)
    else:
        output_img_dir = os.path.join(main_output_path, category, "images", split)
        output_label_dir = os.path.join(main_output_path, category, "labels", split)
        for d in [output_img_dir, output_label_dir]:
            os.makedirs(d, exist_ok=True)

    patch_count_oil = 0
    patch_count_bg = 0
    base_filename = os.path.splitext(image_filename)[0]
    image_path = os.path.join(source_img_dir, image_filename)
    mask_path = os.path.join(source_mask_dir, base_filename + ".png")
    
    if not os.path.exists(mask_path): return 0, 0
        
    try:
        with Image.open(image_path) as img_pil_raw, Image.open(mask_path) as mask_pil_raw:
            img_pil = img_pil_raw.convert("RGB")
            mask_pil = mask_pil_raw.convert("L")
            
            # [v1.4] 雙軌策略：Test 用標準，Train/Val 用自適應
            if split == 'test':
                generated_patches = slice_image_standard_generator(
                    img_pil, mask_pil, patch_size, overlap, 
                    txt_generation_params["target_pixel_value"]
                )
            else:
                generated_patches = slice_image_adaptive_generator(
                    img_pil, mask_pil, patch_size, overlap, 
                    txt_generation_params["target_pixel_value"]
                )
            
            for patch_img, patch_mask, x, y, is_positive in generated_patches:
                # 背景保留率只對非 Test 集生效
                if split != 'test' and not is_positive and (random.random() > effective_bg_keep_ratio):
                    continue

                patch_base_name = f"{base_filename}_patch_x{x}_y{y}"

                if separate_output:
                    current_img_dir, current_label_dir = (output_img_dir_pos, output_label_dir_pos) if is_positive else (output_img_dir_neg, output_label_dir_neg)
                else:
                    current_img_dir, current_label_dir = output_img_dir, output_label_dir
                
                output_img_png_path = os.path.join(current_img_dir, f"{patch_base_name}.png") 
                output_png_path = os.path.join(current_label_dir, f"{patch_base_name}.png")
                output_txt_path = os.path.join(current_label_dir, f"{patch_base_name}.txt")
                
                patch_img.save(output_img_png_path)
                patch_mask.save(output_png_path)
                
                if is_positive:
                    convert_mask_png_to_yolo_txt(output_png_path, output_txt_path, txt_generation_params)
                    patch_count_oil += 1
                else:
                    with open(output_txt_path, 'w') as f: pass 
                    patch_count_bg += 1
                    
    except Exception as e:
        print(f"\n  [ERROR] 處理檔案 '{image_filename}' 時發生嚴重錯誤: {e}")

    return patch_count_oil, patch_count_bg

def main():
    random.seed(RANDOM_SEED)
    
    # [v1.5] 更新輸出資料夾命名，包含兩個 Overlap
    output_dir_name = f"Patched_P{PATCH_SIZE}_Hybrid_TrainO{OVERLAP_TRAIN_VAL}_TestO{OVERLAP_TEST}_BG{int(BACKGROUND_KEEP_RATIO*100)}p"
    if SEPARATE_OUTPUT:
        output_dir_name += "_Separated"
    main_output_path = os.path.join(OUTPUT_BASE_DIR, output_dir_name)
    
    print(f"所有 patch 後的資料將儲存到: {main_output_path}")
    print(f"模式: 雙軌策略 (Hybrid Strategy) [v1.5]")
    print(f"  - Train/Val Overlap: {OVERLAP_TRAIN_VAL} (僅正樣本)")
    print(f"  - Test Overlap     : {OVERLAP_TEST} (標準全圖)")
    
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else (os.cpu_count() or 4)
    print(f"[資訊] 將使用 {num_workers} 個 CPU 核心進行平行處理。")

    current_run_stats = []
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for category in CATEGORIES:
        for split in SPLITS:
            print(f"\n--- 開始處理: Category '{category}', Split '{split}' ---")
            
            source_img_dir = os.path.join(SOURCE_BASE_DIR, category, "images", split)
            source_mask_dir = os.path.join(SOURCE_BASE_DIR, category, "labels", split)
            
            if not os.path.isdir(source_img_dir): continue

            # 建立輸出目錄結構
            if SEPARATE_OUTPUT:
                os.makedirs(os.path.join(main_output_path, category, "images", f"{split}_pos"), exist_ok=True)
                os.makedirs(os.path.join(main_output_path, category, "images", f"{split}_neg"), exist_ok=True)
                os.makedirs(os.path.join(main_output_path, category, "labels", f"{split}_pos"), exist_ok=True)
                os.makedirs(os.path.join(main_output_path, category, "labels", f"{split}_neg"), exist_ok=True)
            else:
                os.makedirs(os.path.join(main_output_path, category, "images", split), exist_ok=True)
                os.makedirs(os.path.join(main_output_path, category, "labels", split), exist_ok=True)

            effective_bg_keep_ratio = BACKGROUND_KEEP_RATIO
            if split == 'test': effective_bg_keep_ratio = 1.0

            image_files = sorted([f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not image_files: continue

            # [v1.5] 根據 Split 選擇對應的重疊量
            if split == 'test':
                current_overlap = OVERLAP_TEST
            else:
                current_overlap = OVERLAP_TRAIN_VAL

            jobs = []
            for image_filename in image_files:
                jobs.append((
                    image_filename, source_img_dir, source_mask_dir, main_output_path,
                    category, split, PATCH_SIZE, current_overlap, effective_bg_keep_ratio,
                    SEPARATE_OUTPUT, TXT_GENERATION_PARAMS
                ))

            total_oil, total_bg = 0, 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(process_single_image, jobs), total=len(jobs), desc=f"Processing {split}"))
            
            for o, b in results:
                total_oil += o
                total_bg += b

            print(f"  結果: 正樣本 {total_oil} 張, 負樣本 {total_bg} 張")
            
            current_run_stats.append({
                "RunTimestamp": run_timestamp, "Category": category, "Split": split,
                "PatchesWithObject": total_oil, "BackgroundPatchesKept": total_bg,
                "TotalPatches": total_oil + total_bg, "Method": "Hybrid_v1.5"
            })

    if current_run_stats:
        excel_log_path = os.path.join(OUTPUT_BASE_DIR, "patch_generation_log.xlsx")
        new_df = pd.DataFrame(current_run_stats)
        if os.path.exists(excel_log_path):
            pd.concat([pd.read_excel(excel_log_path), new_df], ignore_index=True).to_excel(excel_log_path, index=False)
        else:
            new_df.to_excel(excel_log_path, index=False)
            
    print("\n所有任務已完成！")

if __name__ == "__main__":
    main()