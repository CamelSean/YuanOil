# [A] 結構來源資料夾 (Structure Source)
# 這是您的 HNM 產出的資料夾，我們只參考它的「目錄結構」和「檔名清單」。
FOLDER_A_STRUCTURE = r"/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/04_HNM_Experiments/RGB_1024_HNM/hnm_3_gold_dataset_ratio_1_1.5"

# [B] 內容來源資料夾 (Content Source)
# 這是提供實際檔案內容的資料夾 (例如不同的前處理版本，或是原始圖)。
# 它的結構應該要跟 A 類似 (至少要包含 A 裡面有的那些檔案)。
FOLDER_B_CONTENT = r"/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/03_Training_Ready_512/VV_1Ch_1024_Resize_512_Separated"

# [C] 輸出資料夾 (Output Destination)
# 程式會自動建立這個資料夾。
# 結果：C 的結構與檔名會跟 A 一模一樣，但檔案內容是從 B 複製過來的。
FOLDER_C_OUTPUT = r"/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/04_HNM_Experiments/VV_1Ch_1024_HNM_From_RGB"

"""
因為我的實驗有進步，我想要確認到底是資料集的分布導致還是說是資料前處理導致，我想要寫一個code，來將經過run_auto_hnm_cluster_Classifier.py生成出最後的hnm_hybrid_ratio_1_1.5(這邊我設定input folder path A)，
與提供同樣結構的input folder path B，希望生成出output folder C，C裝的是跟A完全一樣的資料夾架構且檔案分布一樣，
不過檔案是由B複製過去的(B會很多檔案，所以要找到相同的檔名才行)，images(png)跟label(png,txt)先要分成一組才行。原始檔案不會影響到

以 A 資料夾為「地圖」，去 B 資料夾「取貨」，然後搬到 C 資料夾。
"""

"""
replicate_structure_replace_content_v2.py (增強版)
功能：以 A 資料夾為架構，從 B 資料夾抓取內容填入 C。
改進：支援 B 資料夾為 Separated 結構 (自動搜尋 _pos / _neg)。
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# === 使用者設定區塊 (User Configuration) ===
# ==============================================================================

# [A] 結構來源資料夾 (Structure Source)
# 這是您的 HNM 產出的資料夾，我們只參考它的「目錄結構」和「檔名清單」。
FOLDER_A_STRUCTURE = r"/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/04_HNM_Experiments/RGB_2048_HNM/hnm_3_gold_dataset_ratio_1_1.5"

# [B] 內容來源資料夾 (Content Source)
# 這是提供實際檔案內容的資料夾 (例如不同的前處理版本，或是原始圖)。
# 它的結構應該要跟 A 類似 (至少要包含 A 裡面有的那些檔案)。
# 結構可以是Separated (例如 images/train_pos, images/train_neg)
FOLDER_B_CONTENT = r"/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/03_Training_Ready_512/VV_1Ch_2048_Resize_512_Separated"

# [C] 輸出資料夾 (Output Destination)
# 程式會自動建立這個資料夾。
# 結果：C 的結構與檔名會跟 A 一模一樣，但檔案內容是從 B 複製過來的。
FOLDER_C_OUTPUT = r"/home/sean/oil_11_26/dataset/DV4_SAR_Small_v3_relabel/04_HNM_Experiments/VV_1Ch_2048_HNM_From_RGB"

# ==============================================================================
# === 核心程式碼 ===
# ==============================================================================

def find_file_in_separated_structure(base_b, relative_path_a):
    """
    在 B 資料夾中尋找檔案，支援 'train' -> 'train_pos' / 'train_neg' 的自動對應。
    """
    # 1. 先嘗試直接對應 (Exact Match)
    direct_path = base_b / relative_path_a
    if direct_path.exists():
        return direct_path

    # 2. 如果找不到，嘗試拆解路徑來搜尋 _pos 或 _neg
    # relative_path_a 例如: "images/train/patch_123.png"
    try:
        parts = list(relative_path_a.parts) # ('images', 'train', 'patch_123.png')
        
        # 我們假設結構通常是 root / type(images/labels) / split(train/val/test) / filename
        if len(parts) >= 3:
            split_name = parts[1] # 'train', 'val', or 'test'
            
            # 嘗試變體 1: split_pos (例如 train_pos)
            parts_pos = list(parts)
            parts_pos[1] = f"{split_name}_pos"
            path_pos = base_b / Path(*parts_pos)
            if path_pos.exists():
                return path_pos
            
            # 嘗試變體 2: split_neg (例如 train_neg)
            parts_neg = list(parts)
            parts_neg[1] = f"{split_name}_neg"
            path_neg = base_b / Path(*parts_neg)
            if path_neg.exists():
                return path_neg

    except Exception:
        pass
        
    return None

def replicate_dataset_content(structure_dir, content_dir, output_dir):
    base_path_a = Path(structure_dir)
    base_path_b = Path(content_dir)
    base_path_c = Path(output_dir)

    print(f"--- 開始執行資料集複製任務 (Smart Mode) ---")
    print(f"1. 結構參考 (A): {base_path_a}")
    print(f"2. 內容來源 (B): {base_path_b}")
    print(f"3. 輸出目標 (C): {base_path_c}")

    if not base_path_a.exists() or not base_path_b.exists():
        print(f"[錯誤] 找不到 A 或 B 資料夾，請檢查路徑。")
        return

    if base_path_c.exists():
        print(f"[警告] 輸出資料夾已存在: {base_path_c} (檔案將被覆蓋)")
    else:
        base_path_c.mkdir(parents=True, exist_ok=True)

    all_files_in_a = [f for f in base_path_a.rglob('*') if f.is_file()]
    
    success_count = 0
    missing_count = 0
    
    print(f"\n正在處理 {len(all_files_in_a)} 個檔案...")

    for file_path_a in tqdm(all_files_in_a, desc="Replicating"):
        # 取得 A 的相對路徑
        relative_path = file_path_a.relative_to(base_path_a)
        
        # --- 關鍵修改：使用智慧搜尋去 B 找檔案 ---
        file_path_b = find_file_in_separated_structure(base_path_b, relative_path)
        
        # 設定 C 的目標路徑 (保持與 A 一樣的結構)
        file_path_c = base_path_c / relative_path
        file_path_c.parent.mkdir(parents=True, exist_ok=True)

        if file_path_b and file_path_b.exists():
            try:
                shutil.copy2(file_path_b, file_path_c)
                success_count += 1
            except Exception as e:
                print(f"[複製失敗] {relative_path}: {e}")
        else:
            missing_count += 1
            # 只有前幾個缺失檔案會印出來，避免洗版
            if missing_count <= 5:
                print(f"[缺失] 在 B 找不到: {relative_path} (已嘗試 _pos/_neg)")

    print("\n" + "="*40)
    print(f"任務完成。成功: {success_count} / 缺失: {missing_count}")
    print(f"輸出位置: {base_path_c}")

if __name__ == "__main__":
    replicate_dataset_content(FOLDER_A_STRUCTURE, FOLDER_B_CONTENT, FOLDER_C_OUTPUT)