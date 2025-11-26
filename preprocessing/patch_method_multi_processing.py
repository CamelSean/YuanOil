# ===================================================================
# ===        preprocessing/patch_method.py (v1.2 - 最終版)        ===
# ===  (包含 v1.1 位移 Bug 修正 + v1.2 多進程加速)            ===
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
import concurrent.futures  # <--- [v1.2] 導入多進程函式庫

# PIL 可能會對大圖有警告，設定此項可避免
Image.MAX_IMAGE_PIXELS = None

# --- [v1.2] CPU 核心數設定 ---
# 設定要使用的 CPU 核心數。None 表示使用所有可用的核心。
# 您可以將其設為一個固定的數字，例如 8 或 16，以保留一些系統資源。
NUM_WORKERS = None
# --- [v1.2 結束] ---

# --- 來源與輸出設定 ---
SOURCE_BASE_DIR = "/home/yuan/Oil_Project_10-8/dataset/datasetv4"
OUTPUT_BASE_DIR = "/home/yuan/Oil_Project_10-8/dataset/datasetv4/DV4_SAR_BIG_v3_relabel_Patch"
CATEGORIES = ["DV4_SAR_BIG_v3_relabel"] # SAR_2 zenodo
SPLITS = ["train", "val", "test"]

# --- Patching 參數設定 ---
PATCH_SIZE = 2048
OVERLAP = 512 # 0 表示不重疊
RANDOM_SEED = 42

# --- 背景樣本保留比例設定 ---
BACKGROUND_KEEP_RATIO = 1

# --- 功能開關 ---
SEPARATE_OUTPUT = True # 是否將正負樣本分開儲存
 
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

def slice_image_with_padding_generator(image_pil, mask_pil, patch_size, overlap):
    """
    生成器函式：對輸入的影像和遮罩進行切割，一次產出一個圖塊 (patch)。
    - [FIX v1.1] 統一使用「左上角 (0,0)」貼上策略，移除了導致位移的「置中」邏輯。
    - 影像填充為白色 (255, 255, 255)，遮罩填充為黑色 (0)。
    """
    image_width, image_height = image_pil.size
    
    pad_color_image = (255, 255, 255) if image_pil.mode == 'RGB' else 255
    pad_value_mask = 0 

    stride = patch_size - overlap
    
    # --- [FIX v1.1] 移除 "if image_width < patch_size..." 的小圖邏輯 ---
    # 下方的 for 迴圈邏輯可以完美處理小圖 (它只會執行一次)
    
    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            
            # 1. 計算實際要從原圖裁切的區域 (不會超過原圖邊界)
            crop_box_actual = (x, y, 
                               min(x + patch_size, image_width), 
                               min(y + patch_size, image_height))
            
            # 2. 裁切影像和遮罩
            region_to_paste_img = image_pil.crop(crop_box_actual)
            region_to_paste_mask = mask_pil.crop(crop_box_actual)
            
            # 3. 建立新的 "畫布" (patch)，影像用白色底，遮罩用黑色底
            patch_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
            patch_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
            
            # 4. [關鍵] 始終將裁切下來的區域貼到 "畫布" 的左上角 (0, 0)
            patch_img.paste(region_to_paste_img, (0, 0))
            patch_seg.paste(region_to_paste_mask, (0, 0))
            
            # 5. 產出這個 "填充" 好的 patch
            #    patch 名稱使用起始座標 (x, y)
            yield patch_img, patch_seg, x, y
    # --- [FIX v1.1 END] ---

def check_target_pixels_exist(mask_segment_pil, target_value):
    """
    檢查一個遮罩圖塊中，是否存在指定的目標像素值。
    """
    segment_array = np.array(mask_segment_pil)
    return np.any(segment_array == target_value)

def convert_mask_png_to_yolo_txt(png_path, output_txt_path, params):
    """
    將二值的 PNG 遮罩圖檔轉換為 YOLOv8 Segmentation 所需的 .txt 標籤格式。
    """
    try:
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: return "error"
        height, width = mask.shape
        if height == 0 or width == 0: return "error"
        
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == params["target_pixel_value"]] = 255
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_format_lines = []
        for contour in contours:
            if cv2.contourArea(contour) < params["min_contour_area"]:
                continue
            
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

# --- [v1.2] 新增多進程處理函式 ---
def process_single_image(job_args):
    """
    [多進程] 處理單一張原始大圖的函式。
    包含：裁切、儲存影像/遮罩、生成 .txt 標籤。
    此函式被設計為在獨立的 CPU 核心上執行。
    """
    # 1. 解包所有參數
    (
        image_filename,
        source_img_dir,
        source_mask_dir,
        main_output_path,
        category,
        split,
        patch_size,
        overlap,
        effective_bg_keep_ratio,
        separate_output,
        txt_generation_params
    ) = job_args

    # 2. 根據 SEPARATE_OUTPUT 的設定，動態決定輸出路徑
    if separate_output:
        output_img_dir_pos = os.path.join(main_output_path, category, "images", f"{split}_pos")
        output_label_dir_pos = os.path.join(main_output_path, category, "labels", f"{split}_pos")
        output_img_dir_neg = os.path.join(main_output_path, category, "images", f"{split}_neg")
        output_label_dir_neg = os.path.join(main_output_path, category, "labels", f"{split}_neg")
        # [多進程安全] 確保目錄存在 (雖然 main 已經建了，但這裡再次檢查)
        os.makedirs(output_img_dir_pos, exist_ok=True); os.makedirs(output_label_dir_pos, exist_ok=True)
        os.makedirs(output_img_dir_neg, exist_ok=True); os.makedirs(output_label_dir_neg, exist_ok=True)
    else:
        output_img_dir = os.path.join(main_output_path, category, "images", split)
        output_label_dir = os.path.join(main_output_path, category, "labels", split)
        os.makedirs(output_img_dir, exist_ok=True); os.makedirs(output_label_dir, exist_ok=True)

    # 3. 複製 `main` 函式中原有的單張圖片處理邏輯
    patch_count_oil = 0
    patch_count_bg = 0
    base_filename = os.path.splitext(image_filename)[0]
    image_path = os.path.join(source_img_dir, image_filename)
    mask_path = os.path.join(source_mask_dir, base_filename + ".png")
    
    if not os.path.exists(mask_path): 
        return patch_count_oil, patch_count_bg
        
    try:
        with Image.open(image_path) as img_pil_raw, Image.open(mask_path) as mask_pil_raw:
            img_pil = img_pil_raw.convert("RGB")
            mask_pil = mask_pil_raw.convert("L")
            generated_patches = slice_image_with_padding_generator(img_pil, mask_pil, patch_size, overlap)
            
            for patch_img, patch_mask, x, y in generated_patches:
                patch_base_name = f"{base_filename}_patch_x{x}_y{y}"
                
                has_target = check_target_pixels_exist(patch_mask, txt_generation_params["target_pixel_value"])
                
                should_save = has_target or (random.random() < effective_bg_keep_ratio)
                if not should_save: 
                    continue

                if separate_output:
                    current_img_dir, current_label_dir = (output_img_dir_pos, output_label_dir_pos) if has_target else (output_img_dir_neg, output_label_dir_neg)
                else:
                    current_img_dir, current_label_dir = output_img_dir, output_label_dir
                
                output_img_png_path = os.path.join(current_img_dir, f"{patch_base_name}.png") 
                output_png_path = os.path.join(current_label_dir, f"{patch_base_name}.png")
                output_txt_path = os.path.join(current_label_dir, f"{patch_base_name}.txt")
                
                patch_img.save(output_img_png_path)
                patch_mask.save(output_png_path)
                
                if has_target:
                    convert_mask_png_to_yolo_txt(output_png_path, output_txt_path, txt_generation_params)
                    patch_count_oil += 1
                else:
                    with open(output_txt_path, 'w') as f: 
                        pass 
                    patch_count_bg += 1
                    
    except Exception as e:
        print(f"\n  [ERROR] 處理檔案 '{image_filename}' 時發生嚴重錯誤: {e}")

    # 4. 回傳此單張圖片產生的 patch 數量
    return patch_count_oil, patch_count_bg
# --- [v1.2 結束] ---


def main():
    """
    主執行函數
    [FIX v1.2] 
    - 修改為使用 concurrent.futures.ProcessPoolExecutor 進行多進程處理。
    - 將單張圖片的處理邏輯移至 `process_single_image` 函式。
    - 在 `main` 中建立工作清單 (jobs) 並行分發任務。
    - 匯總 (aggregate) 來自多進程的回傳結果。
    """
    random.seed(RANDOM_SEED)
    
    # 建立一個唯一的輸出資料夾名稱
    output_dir_name = f"Patched_P{PATCH_SIZE}_O{OVERLAP}_BG{int(BACKGROUND_KEEP_RATIO*100)}p"
    if SEPARATE_OUTPUT:
        output_dir_name += "_Separated"
    main_output_path = os.path.join(OUTPUT_BASE_DIR, output_dir_name)
    
    print(f"所有 patch 後的資料將儲存到: {main_output_path}")
    print(f"使用固定的隨機數種子: {RANDOM_SEED}")
    if SEPARATE_OUTPUT:
        print("[資訊] SEPARATE_OUTPUT 已啟用，正負樣本將被分開儲存。")

    # 初始化日誌記錄所需變數
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_run_stats = []

    # [FIX v1.2] 決定要使用的核心數
    # os.cpu_count() 可能回傳 None，所以我們給一個預設值 4
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else (os.cpu_count() or 4)
    print(f"[資訊] 將使用 {num_workers} 個 CPU 核心進行平行處理。")

    for category in CATEGORIES:
        for split in SPLITS:
            print(f"\n--- 開始處理: Category '{category}', Split '{split}' ---")
            
            source_img_dir = os.path.join(SOURCE_BASE_DIR, category, "images", split)
            source_mask_dir = os.path.join(SOURCE_BASE_DIR, category, "labels", split)
            
            if not os.path.isdir(source_img_dir) or not os.path.isdir(source_mask_dir):
                print(f"  [警告] 來源路徑不存在，跳過: {source_img_dir} 或 {source_mask_dir}")
                continue

            # [FIX v1.2] 建立輸出資料夾的邏輯移到 main，確保在進程開始前已存在
            if SEPARATE_OUTPUT:
                output_img_dir_pos = os.path.join(main_output_path, category, "images", f"{split}_pos")
                output_label_dir_pos = os.path.join(main_output_path, category, "labels", f"{split}_pos")
                os.makedirs(output_img_dir_pos, exist_ok=True); os.makedirs(output_label_dir_pos, exist_ok=True)
                output_img_dir_neg = os.path.join(main_output_path, category, "images", f"{split}_neg")
                output_label_dir_neg = os.path.join(main_output_path, category, "labels", f"{split}_neg")
                os.makedirs(output_img_dir_neg, exist_ok=True); os.makedirs(output_label_dir_neg, exist_ok=True)
            else:
                output_img_dir = os.path.join(main_output_path, category, "images", split)
                output_label_dir = os.path.join(main_output_path, category, "labels", split)
                os.makedirs(output_img_dir, exist_ok=True); os.makedirs(output_label_dir, exist_ok=True)

            effective_bg_keep_ratio = BACKGROUND_KEEP_RATIO
            if split == 'test': 
                effective_bg_keep_ratio = 1.0

            image_files = sorted([f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            source_image_count = len(image_files)
            if source_image_count == 0:
                print("  [資訊] 在此 split 中找不到任何影像，跳過。")
                continue

            # --- [FIX v1.2] 多進程處理 ---
            
            # 1. 建立工作清單 (jobs)
            jobs = []
            for image_filename in image_files:
                job_args = (
                    image_filename,
                    source_img_dir,
                    source_mask_dir,
                    main_output_path,
                    category,
                    split,
                    PATCH_SIZE,
                    OVERLAP,
                    effective_bg_keep_ratio,
                    SEPARATE_OUTPUT,
                    TXT_GENERATION_PARAMS
                )
                jobs.append(job_args)

            # 2. 啟動進程池並執行任務
            total_patches_with_oil = 0
            total_patches_background_kept = 0
            
            print(f"  [資訊] 正在將 {len(jobs)} 張影像分配給 {num_workers} 個核心...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 使用 tqdm 顯示總體進度
                results = list(tqdm(executor.map(process_single_image, jobs), total=len(jobs), desc=f"Processing {category}/{split}", unit="image"))
            
            # 3. 匯總結果
            for patch_count_oil, patch_count_bg in results:
                total_patches_with_oil += patch_count_oil
                total_patches_background_kept += patch_count_bg
            # --- [FIX v1.2 END] ---

            print(f"  處理完成。")
            print(f"  原始圖片數量: {source_image_count}")
            print(f"  儲存了 {total_patches_with_oil} 個包含油汙的 patches。")
            print(f"  保留了 {total_patches_background_kept} 個不含油汙的 patches。")
            
            # 收集該 split 的統計數據到日誌中
            stat_record = {
                "RunTimestamp": run_timestamp,
                "Category": category,
                "Split": split,
                "SourceImageCount": source_image_count,
                "PatchesWithObject": total_patches_with_oil,
                "BackgroundPatchesKept": total_patches_background_kept,
                "TotalPatchesGenerated": total_patches_with_oil + total_patches_background_kept,
                "PatchSize": PATCH_SIZE,
                "Overlap": OVERLAP,
                "BG_Keep_Ratio_Setting": BACKGROUND_KEEP_RATIO,
                "Effective_BG_Keep_Ratio": effective_bg_keep_ratio,
                "RandomSeed": RANDOM_SEED,
                "SEPARATE_OUTPUT_Enabled": SEPARATE_OUTPUT
            }
            current_run_stats.append(stat_record)
    
    # 執行結束後，將日誌寫入 Excel 檔案
    if not current_run_stats:
        print("\n沒有處理任何資料，不產生 Log 檔案。")
        return

    excel_log_path = os.path.join(OUTPUT_BASE_DIR, "patch_generation_log.xlsx")
    new_stats_df = pd.DataFrame(current_run_stats)

    try:
        if os.path.exists(excel_log_path):
            print(f"\n偵測到現有 Log 檔案: {excel_log_path}，將附加新紀錄。")
            existing_df = pd.read_excel(excel_log_path)
            combined_df = pd.concat([existing_df, new_stats_df], ignore_index=True)
        else:
            print(f"\n正在創建新的 Log 檔案: {excel_log_path}")
            combined_df = new_stats_df

        combined_df.to_excel(excel_log_path, index=False)
        print("Excel Log 檔案已成功更新。")
    except Exception as e:
        print(f"[ERROR] 無法寫入 Excel Log 檔案: {e}")
        
    print("\n所有任務已完成！")


if __name__ == "__main__":
    main()