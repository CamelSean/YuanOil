# ======================================================================================
# ===         preprocessing/run_automated_hnm_pipeline_hybrid.py                   ===
# ===  (Hybrid v2.0: VGG Land Clustering + Supervised HNM Mining)                    ===
# ======================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- ⭐⭐⭐ 全域設定 (User Config) ⭐⭐⭐ ---

# 1. 基礎資料夾設定 (沿用原 HNM 設定)
#    請指向包含 images/train_pos, images/train_neg 的根目錄
SOURCE_PATCH_DIR = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/DV4_SAR_Small_v3_relabel_TransferPaperRGB_Fix_Patch/P1024_TrainO512_TestO512_BG100_Split_resize512")

# 2. HNM 流程輸出根目錄
HNM_PIPELINE_BASE_DIR = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/DV4_SAR_Small_v3_relabel_TransferPaperRGB_Fix_Patch/HNM_P1024_TrainO512_TestO512")
# 3. [階段 0] Land 分離設定
VALID_PIXEL_THRESHOLD = 0.2
VGG_PATCH_SIZE = 512

# 4. [階段 1] 分類器訓練設定
CLASSIFIER_MODEL_NAME = 'tf_efficientnetv2_s' # 'mobilenetv3_small_100' 
CLASSIFIER_IMG_SIZE = 512
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 50
CLASSIFIER_LR = 0.0001
CLASSIFIER_LOSS_FUNCTION = 'focal' 
PHASE1_TRAIN_NEG_RATIO = 5

# 5. [階段 2] 探勘設定
HARD_NEGATIVE_THRESHOLD = 0.5   # Prob >= 0.5 -> Hard
MEDIUM_NEGATIVE_THRESHOLD = 0.1 # 0.1 <= Prob < 0.5 -> Medium

# 6. [階段 3] 黃金資料集組裝
TARGET_RATIOS = [1, 1.5, 2, 3] 

# [設定] 初始目標配方：平均分配 (各 25%)
COMPOSITION_RATIO = {
    'land': 0.25,
    'hard': 0.25,
    'medium': 0.25,
    'easy': 0.25
}

RANDOM_SEED = 42

# --- ⭐⭐⭐ 設定結束 ⭐⭐⭐ ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 路徑定義 ---
STAGE0_DIR = HNM_PIPELINE_BASE_DIR / "hnm_0_staging"
CLASSIFIER_MODEL_DIR = HNM_PIPELINE_BASE_DIR / "hnm_1_classifier_model"
CLASSIFIER_MODEL_PATH = CLASSIFIER_MODEL_DIR / "hnm_classifier_hybrid.pt"
CLASSIFIER_PLOT_PATH = CLASSIFIER_MODEL_DIR / "training_curves.png"
MINED_BASE_DIR = HNM_PIPELINE_BASE_DIR / "hnm_2_mined"
MINED_LOG_PATH = HNM_PIPELINE_BASE_DIR / "hnm_2_mining_log.xlsx"
DATASET_STATS_PATH = HNM_PIPELINE_BASE_DIR / "hnm_3_gold_dataset_stats.xlsx"

# ======================================================================================
# === 輔助函式
# ======================================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1-pt)**self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def plot_training_history(history, save_path):
    ep = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1); plt.plot(ep, history['train_loss'], 'b-', label='Train'); plt.plot(ep, history['val_loss'], 'r-', label='Val'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2); plt.plot(ep, history['val_acc'], 'g-', label='Val Acc'); plt.legend(); plt.grid(True)
    plt.savefig(save_path); plt.close()

def ensure_clean_dir(d):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# ======================================================================================
# === Stage 0: VGG Land Separation
# ======================================================================================
def get_vgg_extractor():
    w = models.VGG16_Weights.IMAGENET1K_V1
    m = models.vgg16(weights=w).features
    m = nn.Sequential(m, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()).to(device).eval()
    return m

def get_valid_crop_area(cv2_img):
    if cv2_img is None: return None
    if len(cv2_img.shape) == 3: gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    else: gray = cv2_img
    mask = (gray > 5) & (gray < 250)
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, w, h)

def extract_features_vgg(model, img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None: return None
        rect = get_valid_crop_area(img)
        if not rect: return None
        x, y, w, h = rect
        crop = img[y:y+h, x:x+w]
        
        if crop.shape[0] < VGG_PATCH_SIZE or crop.shape[1] < VGG_PATCH_SIZE:
            crop = cv2.resize(crop, (VGG_PATCH_SIZE, VGG_PATCH_SIZE))
        else:
            h, w = crop.shape[:2]
            ry = np.random.randint(0, h - VGG_PATCH_SIZE + 1)
            rx = np.random.randint(0, w - VGG_PATCH_SIZE + 1)
            crop = crop[ry:ry+VGG_PATCH_SIZE, rx:rx+VGG_PATCH_SIZE]
            
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(Image.fromarray(crop)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            return model(t).cpu().numpy().flatten()
    except: return None

def process_split_clustering(split_name, model):
    src_dir = SOURCE_PATCH_DIR / "images" / f"{split_name}_neg"
    src_lbl = SOURCE_PATCH_DIR / "labels" / f"{split_name}_neg"
    if not src_dir.exists(): return

    dirs = {
        'land': (STAGE0_DIR / split_name / "land" / "images", STAGE0_DIR / split_name / "land" / "labels"),
        'water': (STAGE0_DIR / split_name / "water" / "images", STAGE0_DIR / split_name / "water" / "labels")
    }
    for d in dirs.values(): d[0].mkdir(parents=True, exist_ok=True); d[1].mkdir(parents=True, exist_ok=True)
    
    files = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg"))
    logging.info(f"--- 分析 {split_name}_neg ({len(files)} 張) ---")
    
    valid_files, feats = [], []
    for f in tqdm(files, desc="VGG Features"):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if np.count_nonzero((img>5)&(img<250)) / img.size < VALID_PIXEL_THRESHOLD: continue
        
        ft = extract_features_vgg(model, f)
        if ft is not None:
            valid_files.append(f)
            feats.append(ft)
            
    if not valid_files: return

    logging.info("  - K-Means 分群...")
    X = np.array(feats)
    pca = PCA(n_components=min(50, len(X)), random_state=42)
    X_pca = pca.fit_transform(X)
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_pca)
    
    # 找 Land (Std 最高)
    cluster_stds = {}
    for cid in range(3):
        indices = np.where(labels == cid)[0]
        samp = [valid_files[i] for i in np.random.choice(indices, min(len(indices), 50), replace=False)]
        stds = []
        for s in samp:
            im = cv2.imread(str(s), cv2.IMREAD_GRAYSCALE)
            mask = (im > 5) & (im < 250)
            if mask.sum() > 0: stds.append(np.std(im[mask]))
        cluster_stds[cid] = np.mean(stds) if stds else 0
        
    land_cid = max(cluster_stds, key=cluster_stds.get)
    
    counts = {'land': 0, 'water': 0}
    for idx, cid in enumerate(labels):
        f = valid_files[idx]
        cat = 'land' if cid == land_cid else 'water'
        counts[cat] += 1
        
        tgt_img, tgt_lbl = dirs[cat]
        shutil.copy(f, tgt_img / f.name)
        for ext in ['.txt', '.png']:
            l = src_lbl / (f.stem + ext)
            if l.exists(): shutil.copy(l, tgt_lbl / l.name)
            
    logging.info(f"  - 分離結果: Land={counts['land']}, Water={counts['water']}")

def run_phase_0():
    logging.info("=== Phase 0: VGG Land Separation ===")
    ensure_clean_dir(STAGE0_DIR)
    m = get_vgg_extractor()
    process_split_clustering('train', m)
    process_split_clustering('val', m)

# ======================================================================================
# === Stage 1: Train Classifier
# ======================================================================================
class ClassifierDS(Dataset):
    def __init__(self, samples, tf): self.samples = samples; self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if self.tf: im = self.tf(image=im)['image']
        return im, torch.tensor(l, dtype=torch.long)

def get_transforms(s):
    # [FIX] Changed ShiftScaleRotate to Affine to avoid warnings
    base = [A.Resize(s, s), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
    aug = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), p=0.5)]
    return {'train': A.Compose(aug + base), 'val': A.Compose(base)}

def prep_samples(split):
    pos = list((SOURCE_PATCH_DIR / "images" / f"{split}_pos").glob("*"))
    neg = list((STAGE0_DIR / split / "water" / "images").glob("*"))
    if not pos or not neg: return []
    
    n_neg = int(len(pos) * PHASE1_TRAIN_NEG_RATIO)
    sel_neg = random.sample(neg, n_neg) if len(neg) > n_neg else neg
    s = [(p, 1) for p in pos] + [(p, 0) for p in sel_neg]
    random.shuffle(s)
    return s

def run_phase_1():
    logging.info("=== Phase 1: Train Classifier ===")
    ensure_clean_dir(CLASSIFIER_MODEL_DIR)
    
    tr_s = prep_samples('train')
    val_s = prep_samples('val')
    tf = get_transforms(CLASSIFIER_IMG_SIZE)
    
    train_ds = ClassifierDS(tr_s, tf['train'])
    val_ds = ClassifierDS(val_s, tf['val'])
    
    # [FIX] pin_memory=False to avoid potential CUDA issues on some systems
    tr_l = DataLoader(train_ds, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
    val_l = DataLoader(val_ds, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
    
    m = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=True, num_classes=2).to(device)
    opt = optim.AdamW(m.parameters(), lr=CLASSIFIER_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    crit = FocalLoss().to(device)
    
    best_acc = 0
    hist = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for ep in range(CLASSIFIER_EPOCHS):
        m.train(); rl = 0
        for x, y in tqdm(tr_l, desc=f"Ep {ep+1}"):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                loss = crit(m(x), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            rl += loss.item() * x.size(0)
            
        m.eval(); vl = 0; corr = 0
        with torch.no_grad():
            for x, y in val_l:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    out = m(x); vl += crit(out, y).item() * x.size(0)
                corr += (out.argmax(1) == y).sum().item()
        
        acc = corr / len(val_s)
        hist['train_loss'].append(rl/len(tr_s)); hist['val_loss'].append(vl/len(val_s)); hist['val_acc'].append(acc)
        if acc > best_acc: best_acc = acc; torch.save(m.state_dict(), CLASSIFIER_MODEL_PATH)
        logging.info(f"  Acc: {acc:.4f}")
        
    plot_training_history(hist, CLASSIFIER_PLOT_PATH)

# ======================================================================================
# === Stage 2: Mining
# ======================================================================================
class MineDS(Dataset):
    def __init__(self, files, tf): self.files = files; self.tf = tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        p = self.files[i]
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        return self.tf(image=im)['image'], str(p)

def copy_labeled(src, dst_base, split):
    base = Path(src).stem
    lbl_src = STAGE0_DIR / split / "water" / "labels"
    d_i = dst_base / "images"; d_l = dst_base / "labels"
    d_i.mkdir(parents=True, exist_ok=True); d_l.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(src, d_i / Path(src).name)
    for e in ['.txt', '.png']:
        l = lbl_src / (base + e)
        if l.exists(): shutil.copy(l, d_l / l.name)

def run_phase_2():
    logging.info("=== Phase 2: Mining ===")
    m = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=False, num_classes=2).to(device)
    m.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device)); m.eval()
    tf = get_transforms(CLASSIFIER_IMG_SIZE)['val']
    
    logs = []
    for split in ['train', 'val']:
        src = STAGE0_DIR / split / "water" / "images"
        if not src.exists(): continue
        
        files = list(src.glob("*.png")) + list(src.glob("*.jpg"))
        loader = DataLoader(MineDS(files, tf), batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
        
        root = MINED_BASE_DIR / split; ensure_clean_dir(root)
        cnt = {'hard':0, 'medium':0, 'easy':0}
        
        with torch.no_grad():
            for imgs, paths in tqdm(loader, desc=f"Mining {split}"):
                probs = torch.softmax(m(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
                for p, path in zip(probs, paths):
                    if p >= HARD_NEGATIVE_THRESHOLD: c = 'hard'
                    elif p >= MEDIUM_NEGATIVE_THRESHOLD: c = 'medium'
                    else: c = 'easy'
                    
                    cnt[c] += 1
                    copy_labeled(path, root / c, split)
                    logs.append({'split': split, 'file': Path(path).name, 'prob': p, 'cat': c})
        logging.info(f"  {split}: {cnt}")
    pd.DataFrame(logs).to_excel(MINED_LOG_PATH, index=False)

# ======================================================================================
# === Stage 3: Assembly (Priority Filling)
# ======================================================================================
def copy_final_batch(src_root, dst_img_dir, dst_lbl_dir, stems):
    """
    [修正]: 明確接受目標的圖片與標籤資料夾，不再自動串接 /images
    """
    s_i = src_root / "images"; s_l = src_root / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    c = 0
    for s in stems:
        f = s_i / f"{s}.png"
        if not f.exists(): f = s_i / f"{s}.jpg"
        if f.exists():
            shutil.copy(f, dst_img_dir / f.name)
            for e in ['.txt', '.png']:
                l = s_l / (s + e)
                if l.exists(): shutil.copy(l, dst_lbl_dir / l.name)
            c += 1
    return c

def assemble_split(ratio, split, out_dir):
    pos_dir = SOURCE_PATCH_DIR / "images" / f"{split}_pos"
    land_root = STAGE0_DIR / split / "land"
    mined_root = MINED_BASE_DIR / split
    
    # 1. 獲取所有可用檔案
    def get_stems(d): return [f.stem for f in (d/"images").glob("*")] if (d/"images").exists() else []
    
    pos_files = get_stems(SOURCE_PATCH_DIR) # Placeholder
    pos_files = [f.stem for f in pos_dir.glob("*")] # Real pos files
    
    land_files = get_stems(land_root)
    hard_files = get_stems(mined_root / "hard")
    med_files = get_stems(mined_root / "medium")
    easy_files = get_stems(mined_root / "easy")
    
    n_pos = len(pos_files)
    n_neg_total = int(n_pos * ratio)
    
    # 2. 計算初始目標 (各 25%)
    t_land = int(n_neg_total * COMPOSITION_RATIO['land'])
    t_hard = int(n_neg_total * COMPOSITION_RATIO['hard'])
    t_med = int(n_neg_total * COMPOSITION_RATIO['medium'])
    t_easy = int(n_neg_total * COMPOSITION_RATIO['easy'])
    
    # 3. 初始抽樣
    sel_land = random.sample(land_files, min(len(land_files), t_land))
    sel_hard = random.sample(hard_files, min(len(hard_files), t_hard))
    sel_med = random.sample(med_files, min(len(med_files), t_med))
    sel_easy = random.sample(easy_files, min(len(easy_files), t_easy))
    
    current_count = len(sel_land) + len(sel_hard) + len(sel_med) + len(sel_easy)
    shortage = n_neg_total - current_count
    
    # 4. [優先級填補] Hard > Medium > Land > Easy
    if shortage > 0:
        logging.info(f"  [{split}] 初始樣本不足 (缺 {shortage})，啟動優先級填補...")
        fill_order = [
            ('hard', hard_files, sel_hard),
            ('medium', med_files, sel_med),
            ('land', land_files, sel_land),
            ('easy', easy_files, sel_easy) 
        ]
        
        for name, all_files, selected in fill_order:
            if shortage <= 0: break
            remaining = list(set(all_files) - set(selected))
            if not remaining: continue
            
            take = min(len(remaining), shortage)
            sel = random.sample(remaining, take)
            selected.extend(sel)
            shortage -= take
            logging.info(f"    -> 從 {name} 補充 {take} 張")
            
    # 5. 執行複製 (修正為正確的 YOLO 結構)
    
    # 定義目標資料夾 (確保 images 和 labels 在根目錄)
    d_img = out_dir / "images" / split
    d_lbl = out_dir / "labels" / split
    d_img.mkdir(parents=True, exist_ok=True)
    d_lbl.mkdir(parents=True, exist_ok=True)
    
    # 複製正樣本
    c_pos = 0
    for s in pos_files:
        f = pos_dir / f"{s}.png"
        if not f.exists(): f = pos_dir / f"{s}.jpg"
        shutil.copy(f, d_img / f.name)
        for e in ['.txt', '.png']:
            l = SOURCE_PATCH_DIR / "labels" / f"{split}_pos" / (s+e)
            if l.exists(): shutil.copy(l, d_lbl / l.name)
        c_pos += 1
        
    # 複製負樣本 (傳入明確的目標資料夾)
    c_land = copy_final_batch(land_root, d_img, d_lbl, sel_land)
    c_hard = copy_final_batch(mined_root / "hard", d_img, d_lbl, sel_hard)
    c_med = copy_final_batch(mined_root / "medium", d_img, d_lbl, sel_med)
    c_easy = copy_final_batch(mined_root / "easy", d_img, d_lbl, sel_easy)
    
    return {'pos': c_pos, 'land': c_land, 'hard': c_hard, 'medium': c_med, 'easy': c_easy, 'total_neg': c_land+c_hard+c_med+c_easy}

def run_phase_3():
    logging.info("=== Phase 3: Assembly ===")
    stats = []
    for r in TARGET_RATIOS:
        out = HNM_PIPELINE_BASE_DIR / f"hnm_hybrid_ratio_1_{r}"
        ensure_clean_dir(out)
        
        logging.info(f"--- Ratio 1:{r} ---")
        st_tr = assemble_split(r, 'train', out)
        st_val = assemble_split(r, 'val', out)
        
        # Test Set (Copy all)
        t_i = out / "images" / "test"; t_i.mkdir(parents=True, exist_ok=True)
        t_l = out / "labels" / "test"; t_l.mkdir(parents=True, exist_ok=True)
        for suf in ['_pos', '_neg']:
            s_i = SOURCE_PATCH_DIR / "images" / f"test{suf}"
            s_l = SOURCE_PATCH_DIR / "labels" / f"test{suf}"
            if s_i.exists():
                for f in s_i.glob("*"): shutil.copy(f, t_i / f.name)
                for f in s_l.glob("*"): shutil.copy(f, t_l / f.name)
                
        stats.append({
            'Ratio': f"1:{r}",
            'Tr_Pos': st_tr['pos'], 'Tr_Neg': st_tr['total_neg'],
            'Tr_Land': st_tr['land'], 'Tr_Hard': st_tr['hard'], 'Tr_Med': st_tr['medium'], 'Tr_Easy': st_tr['easy'],
            'Val_Pos': st_val['pos'], 'Val_Neg': st_val['total_neg'],
            'Out': str(out)
        })
    pd.DataFrame(stats).to_excel(DATASET_STATS_PATH, index=False)

if __name__ == "__main__":
    try:
        run_phase_0()
        run_phase_1()
        run_phase_2()
        run_phase_3()
        logging.info("Done.")
    except Exception as e:
        logging.exception(f"Error: {e}")