# ======================================================================================
# ===         preprocessing/run_automated_hnm_pipeline.py                          ===
# ===  (v1.7: 新增訓練過程繪圖 (Loss/Acc) 與 最終資料集統計 Excel 報表)              ===
# ======================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
import logging
import matplotlib.pyplot as plt  

# --- ⭐⭐⭐ 全域設定 (您唯一需要修改的地方) ⭐⭐⭐ ---

# 1. 基礎資料夾設定
SOURCE_PATCH_DIR = Path("/home/yuan/Oil_Project_10-8/dataset/datasetv4/DV4_SAR_BIG_v3_relabel_Patch/Patched_P1024_O512_BG100p_Separated/DV4_SAR_BIG_v3_relabel_resize512")

# 2. HNM 流程輸出根目錄
HNM_PIPELINE_BASE_DIR = Path("/home/yuan/Oil_Project_10-8/dataset/datasetv4/DV4_SAR_BIG_v3_relabel_Patch/HNM_P1024_O512_Resize512")

# 3. 階段 1：分類器訓練設定
CLASSIFIER_MODEL_NAME = 'mobilenetv3_small_100' 
CLASSIFIER_IMG_SIZE = 512
CLASSIFIER_BATCH_SIZE = 128
CLASSIFIER_EPOCHS = 100
CLASSIFIER_LR = 0.0001
CLASSIFIER_LOSS_FUNCTION = 'focal' 

# [v1.4/v1.6] 階段 1 訓練時的最大正負樣本比例 (1 : N)
PHASE1_TRAIN_NEG_RATIO = 5 

# 4. 階段 2：困難樣本探勘設定 (三元分割)
HARD_NEGATIVE_THRESHOLD = 0.5   # Prob >= 0.5   -> Hard (全部保留)
MEDIUM_NEGATIVE_THRESHOLD = 0.1 # Prob >= 0.1   -> Medium

# 5. 階段 3：黃金資料集組裝設定 (多重比例)
TARGET_RATIOS = [1, 2, 3] 

REMAINING_FILL_RATIO = {
    'medium': 0.7,  
    'easy': 0.3     
}

RANDOM_SEED = 42

# --- ⭐⭐⭐ 設定結束，以下為腳本主體 ⭐⭐⭐ ---

# --- 通用設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 輸出路徑定義 ---
CLASSIFIER_MODEL_DIR = HNM_PIPELINE_BASE_DIR / "hnm_1_classifier_model"
CLASSIFIER_MODEL_PATH = CLASSIFIER_MODEL_DIR / "hnm_classifier_v1_best.pt"
CLASSIFIER_PLOT_PATH = CLASSIFIER_MODEL_DIR / "training_curves.png" 

# 階段 2 輸出路徑
MINED_BASE_DIR = HNM_PIPELINE_BASE_DIR / "hnm_2_mined"
MINED_LOG_PATH = HNM_PIPELINE_BASE_DIR / "hnm_2_mining_log.xlsx"

# 階段 3 統計報表路徑
DATASET_STATS_PATH = HNM_PIPELINE_BASE_DIR / "hnm_3_gold_dataset_stats.xlsx"

# ======================================================================================
# === 輔助類別
# ======================================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

class PatchClassifierDataset(Dataset):
    def __init__(self, samples_list, transform=None):
        self.transform = transform
        self.samples = samples_list
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None: raise IOError
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if self.transform: image = self.transform(image=image)['image']
            return image, torch.tensor(label, dtype=torch.long)
        except: return torch.zeros((3, CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE)), torch.tensor(0, dtype=torch.long)

def get_classifier_transforms(img_size):
    return {
        'train': A.Compose([
            A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(img_size, img_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()
        ])
    }

def plot_training_history(history, save_path):
    """繪製並儲存訓練曲線"""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Curve'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy')
    plt.title('Validation Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"訓練曲線圖已儲存至: {save_path}")

# ======================================================================================
# === 階段 1：訓練分類器 (Model v1)
# ======================================================================================
def prepare_classifier_samples(split_name):
    pos_dir = SOURCE_PATCH_DIR / "images" / f"{split_name}_pos"
    neg_dir = SOURCE_PATCH_DIR / "images" / f"{split_name}_neg"
    
    if not pos_dir.is_dir() or not neg_dir.is_dir():
        logging.error(f"找不到 {split_name} 資料夾：{pos_dir} 或 {neg_dir}")
        return []

    pos_files = list(pos_dir.glob("*.png")) + list(pos_dir.glob("*.jpg"))
    neg_files = list(neg_dir.glob("*.png")) + list(neg_dir.glob("*.jpg"))
    
    target_neg_count = int(len(pos_files) * PHASE1_TRAIN_NEG_RATIO)
    
    if len(neg_files) > target_neg_count:
        logging.info(f"  [{split_name}] 負樣本抽樣: {len(neg_files)} -> {target_neg_count} (Ratio 1:{PHASE1_TRAIN_NEG_RATIO})")
        selected_neg_files = random.sample(neg_files, target_neg_count)
    else:
        logging.info(f"  [{split_name}] 使用全部負樣本: {len(neg_files)}")
        selected_neg_files = neg_files
    
    samples = []
    for p in pos_files: samples.append((p, 1))
    for p in selected_neg_files: samples.append((p, 0))
    random.shuffle(samples)
    return samples

def run_phase_1_train_classifier():
    logging.info("="*60)
    logging.info("=== HNM 階段 1：訓練分類器 (Strict Split Mode) ===")
    logging.info("="*60)
    
    CLASSIFIER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    train_samples = prepare_classifier_samples('train')
    val_samples = prepare_classifier_samples('val')
    
    logging.info(f"分類器資料集準備完成：")
    logging.info(f"  - Train Set: {len(train_samples)} samples")
    logging.info(f"  - Val Set  : {len(val_samples)} samples")
    
    transforms = get_classifier_transforms(CLASSIFIER_IMG_SIZE)
    train_dataset = PatchClassifierDataset(train_samples, transform=transforms['train'])
    val_dataset = PatchClassifierDataset(val_samples, transform=transforms['val'])

    # [FIX] pin_memory=False 避免 CUDA error
    train_loader = DataLoader(train_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
    
    model = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=True, num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CLASSIFIER_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    if CLASSIFIER_LOSS_FUNCTION.lower() == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CLASSIFIER_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer) # [FIX] 傳入 optimizer
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_dataset)
        
        model.eval()
        val_loss = 0.0; corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Val"):
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
        
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = corrects.double() / len(val_dataset)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        
        logging.info(f"Epoch {epoch+1} -> Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), CLASSIFIER_MODEL_PATH)
            logging.info(f"*** New Best Model Saved (Acc: {best_val_acc:.4f}) ***")

    plot_training_history(history, CLASSIFIER_PLOT_PATH)

# ======================================================================================
# === 階段 2：探勘 (Mining) 
# ======================================================================================
class MiningDataset(Dataset):
    def __init__(self, neg_dir, transform=None):
        self.transform = transform
        self.samples = list(neg_dir.glob("*.png")) + list(neg_dir.glob("*.jpg"))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None: raise IOError
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if self.transform: image = self.transform(image=image)['image']
            return image, str(img_path)
        except: return torch.zeros((3, CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE)), str(img_path)

def copy_mined_file(src_path, dest_dir, base_name, split_name):
    src_path = Path(src_path)
    src_label_txt = SOURCE_PATCH_DIR / "labels" / f"{split_name}_neg" / f"{base_name}.txt"
    src_label_png = SOURCE_PATCH_DIR / "labels" / f"{split_name}_neg" / f"{base_name}.png"
    
    dest_img_dir = dest_dir / "images"
    dest_label_dir = dest_dir / "labels"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_label_dir.mkdir(parents=True, exist_ok=True)
    
    if src_path.exists(): shutil.copy(src_path, dest_img_dir / src_path.name)
    if src_label_txt.exists(): shutil.copy(src_label_txt, dest_label_dir / src_label_txt.name)
    if src_label_png.exists(): shutil.copy(src_label_png, dest_label_dir / src_label_png.name)

def mine_specific_split(model, split_name, results_log):
    logging.info(f"--- 正在探勘 {split_name}_neg ---")
    
    neg_dir = SOURCE_PATCH_DIR / "images" / f"{split_name}_neg"
    if not neg_dir.exists():
        logging.warning(f"找不到 {neg_dir}，跳過此 split。")
        return

    base_out = MINED_BASE_DIR / split_name
    dir_hard = base_out / "hard"; dir_medium = base_out / "medium"; dir_easy = base_out / "easy"
    for d in [dir_hard, dir_medium, dir_easy]: 
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    transforms = get_classifier_transforms(CLASSIFIER_IMG_SIZE)['val']
    dataset = MiningDataset(neg_dir, transform=transforms)
    loader = DataLoader(dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
    
    counts = {'Hard': 0, 'Medium': 0, 'Easy': 0}
    
    with torch.no_grad():
        for inputs, img_paths in tqdm(loader, desc=f"Mining {split_name}"):
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                prob_pos = probs[:, 1].cpu().numpy()
            
            for i, path_str in enumerate(img_paths):
                prob = prob_pos[i]
                name = Path(path_str).stem
                
                if prob >= HARD_NEGATIVE_THRESHOLD: status = "Hard"; dest = dir_hard
                elif prob >= MEDIUM_NEGATIVE_THRESHOLD: status = "Medium"; dest = dir_medium
                else: status = "Easy"; dest = dir_easy
                
                copy_mined_file(path_str, dest, name, split_name)
                counts[status] += 1
                
                results_log.append({"split": split_name, "file": name, "prob": f"{prob:.4f}", "status": status})
    
    logging.info(f"[{split_name}] 結果: Hard={counts['Hard']}, Medium={counts['Medium']}, Easy={counts['Easy']}")

def run_phase_2_mine_negatives():
    logging.info("="*60)
    logging.info("=== HNM 階段 2：執行分類器探勘 (Strict Mode) ===")
    logging.info("="*60)
    
    model = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    model.eval()
    
    results_log = []
    mine_specific_split(model, 'train', results_log)
    mine_specific_split(model, 'val', results_log)
    
    df = pd.DataFrame(results_log)
    df.to_excel(MINED_LOG_PATH, index=False)
    logging.info(f"詳細日誌已儲存至：{MINED_LOG_PATH}")

# ======================================================================================
# === 階段 3：組裝黃金資料集 (Multi-Ratio, Strict Split) - 包含詳細統計
# ======================================================================================

def copy_dataset_split(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, file_stems):
    """大量複製檔案並回傳成功複製的數量"""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for stem in file_stems:
        src_img = src_img_dir / f"{stem}.png"
        if not src_img.exists(): src_img = src_img_dir / f"{stem}.jpg"
        
        src_lbl_txt = src_lbl_dir / f"{stem}.txt"
        src_lbl_png = src_lbl_dir / f"{stem}.png"
        
        if src_img.exists():
            shutil.copy(src_img, dst_img_dir / src_img.name)
            if src_lbl_txt.exists(): shutil.copy(src_lbl_txt, dst_lbl_dir / src_lbl_txt.name)
            if src_lbl_png.exists(): shutil.copy(src_lbl_png, dst_lbl_dir / src_lbl_png.name)
            count += 1
    return count

def assemble_single_split(ratio, split_name, output_root):
    """
    組裝單一個 split (train/val)，回傳 (pos_count, neg_count)
    """
    pos_img_dir = SOURCE_PATCH_DIR / "images" / f"{split_name}_pos"
    pos_lbl_dir = SOURCE_PATCH_DIR / "labels" / f"{split_name}_pos"
    
    mined_base = MINED_BASE_DIR / split_name
    neg_hard_dir = mined_base / "hard" / "images"
    neg_med_dir = mined_base / "medium" / "images"
    neg_easy_dir = mined_base / "easy" / "images"
    
    pos_files = [p.stem for p in pos_img_dir.glob("*")]
    hard_files = [p.stem for p in neg_hard_dir.glob("*")]
    med_files = [p.stem for p in neg_med_dir.glob("*")]
    easy_files = [p.stem for p in neg_easy_dir.glob("*")]
    
    n_pos = len(pos_files)
    n_neg_target = int(n_pos * ratio)
    
    # 篩選負樣本
    selected_neg = list(hard_files) 
    
    quota_remaining = n_neg_target - len(selected_neg)
    if quota_remaining > 0:
        n_med_target = int(quota_remaining * REMAINING_FILL_RATIO['medium'])
        n_easy_target = quota_remaining - n_med_target
        
        if len(med_files) >= n_med_target: selected_neg.extend(random.sample(med_files, n_med_target))
        else: selected_neg.extend(med_files)
        
        if len(easy_files) >= n_easy_target: selected_neg.extend(random.sample(easy_files, n_easy_target))
        else: selected_neg.extend(easy_files)
    
    logging.info(f"  [{split_name}] Pos: {n_pos}, Neg Target: {n_neg_target}, Selected Neg: {len(selected_neg)}")

    dst_img = output_root / "images" / split_name
    dst_lbl = output_root / "labels" / split_name
    
    # 執行複製
    real_pos_count = copy_dataset_split(pos_img_dir, pos_lbl_dir, dst_img, dst_lbl, pos_files)
    
    neg_hard_sel = [f for f in selected_neg if f in hard_files]
    neg_med_sel = [f for f in selected_neg if f in med_files]
    neg_easy_sel = [f for f in selected_neg if f in easy_files]
    
    c1 = copy_dataset_split(mined_base / "hard" / "images", mined_base / "hard" / "labels", dst_img, dst_lbl, neg_hard_sel)
    c2 = copy_dataset_split(mined_base / "medium" / "images", mined_base / "medium" / "labels", dst_img, dst_lbl, neg_med_sel)
    c3 = copy_dataset_split(mined_base / "easy" / "images", mined_base / "easy" / "labels", dst_img, dst_lbl, neg_easy_sel)
    
    return real_pos_count, c1 + c2 + c3

def run_phase_3_assemble_gold_dataset_multi_ratio():
    logging.info("="*60)
    logging.info("=== HNM 階段 3：開始組裝多組黃金資料集 (Strict Split) ===")
    logging.info("="*60)
    
    all_stats = [] # 用於收集統計數據
    
    for ratio in TARGET_RATIOS:
        current_ratio_name = f"1_{ratio}"
        output_dir = HNM_PIPELINE_BASE_DIR / f"hnm_3_gold_dataset_ratio_{current_ratio_name}"
        if output_dir.exists(): shutil.rmtree(output_dir)
        
        logging.info(f"--- 正在組裝 Ratio 1:{ratio} ---")
        
        # 1. 組裝 Train & Val
        train_pos, train_neg = assemble_single_split(ratio, 'train', output_dir)
        val_pos, val_neg = assemble_single_split(ratio, 'val', output_dir)
        
        # 2. 複製 Test (原封不動)
        logging.info("  [test] 複製測試集 (原封不動)...")
        
        dst_test_img = output_dir / "images" / "test"
        dst_test_lbl = output_dir / "labels" / "test"
        dst_test_img.mkdir(parents=True, exist_ok=True)
        dst_test_lbl.mkdir(parents=True, exist_ok=True)

        test_pos_count = 0
        test_neg_count = 0

        # 嘗試從 _pos / _neg 複製 (v1.5 patch method 格式)
        for suffix in ["_pos", "_neg"]:
            src_img = SOURCE_PATCH_DIR / "images" / f"test{suffix}"
            src_lbl = SOURCE_PATCH_DIR / "labels" / f"test{suffix}"
            if src_img.exists():
                file_count = 0
                for f in src_img.glob("*"): 
                    shutil.copy(f, dst_test_img / f.name)
                    file_count += 1
                for f in src_lbl.glob("*"): shutil.copy(f, dst_test_lbl / f.name)
                
                if suffix == "_pos": test_pos_count = file_count
                else: test_neg_count = file_count
        
        # 舊版相容
        if test_pos_count == 0 and test_neg_count == 0:
             src_img_flat = SOURCE_PATCH_DIR / "images" / "test"
             src_lbl_flat = SOURCE_PATCH_DIR / "labels" / "test"
             if src_img_flat.exists():
                 for f in src_img_flat.glob("*"): shutil.copy(f, dst_test_img / f.name)
                 for f in src_lbl_flat.glob("*"): shutil.copy(f, dst_test_lbl / f.name)
                 test_pos_count = len(list(src_img_flat.glob("*"))) # 無法區分，暫記為 pos 或 total

        logging.info(f"  資料集已建立於: {output_dir}")
        
        # 收集統計數據
        stats = {
            'Ratio_Setting': f"1:{ratio}",
            'Train_Pos': train_pos,
            'Train_Neg': train_neg,
            'Train_Total': train_pos + train_neg,
            'Train_Actual_Ratio': f"1:{train_neg/train_pos:.2f}" if train_pos > 0 else "N/A",
            'Val_Pos': val_pos,
            'Val_Neg': val_neg,
            'Val_Total': val_pos + val_neg,
            'Val_Actual_Ratio': f"1:{val_neg/val_pos:.2f}" if val_pos > 0 else "N/A",
            'Test_Pos': test_pos_count,
            'Test_Neg': test_neg_count,
            'Test_Total': test_pos_count + test_neg_count,
            'Output_Path': str(output_dir)
        }
        all_stats.append(stats)

    # 輸出 Excel 報表
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_excel(DATASET_STATS_PATH, index=False)
    logging.info(f"資料集統計報表已儲存至: {DATASET_STATS_PATH}")

# ======================================================================================
# === 主執行流程
# ======================================================================================
def main():
    logging.info(f"HNM 自動化管線 (v1.8 Final) 已啟動。")
    
    try:
        if CLASSIFIER_MODEL_PATH.exists():
            logging.info("偵測到舊分類器，重新訓練以確保設定一致...")
            shutil.rmtree(CLASSIFIER_MODEL_PATH.parent)
            
        run_phase_1_train_classifier()
        run_phase_2_mine_negatives()
        run_phase_3_assemble_gold_dataset_multi_ratio()
        
        logging.info("\n🎉 所有任務完成！")
        
    except Exception as e:
        logging.exception(f"執行失敗：{e}")

if __name__ == "__main__":
    main()