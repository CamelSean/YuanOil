import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import logging
import cv2

def parse_scene_id(filename):
    """
    從檔名解析場景 ID，用於避免 Data Leakage。
    假設格式: S1A_..._x1024_y512.png
    邏輯: 取最後一個 "_x" 之前的所有字串作為 ID。
    """
    filename = str(filename)
    if "_x" in filename:
        return filename.rsplit("_x", 1)[0]
    return filename

def get_sample_class(label_path):
    """
    讀取標籤內容以判斷是正樣本(1)還是負樣本(0)。
    """
    if not label_path.exists():
        return 0
    
    # 針對 YOLO .txt 格式
    if label_path.suffix == '.txt':
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            return 1 if len(lines) > 0 else 0
            
    # 針對 Mask .png 格式
    elif label_path.suffix in ['.png', '.jpg']:
        try:
            mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if mask is None: return 0
            return 1 if cv2.countNonZero(mask) > 0 else 0
        except:
            return 0
    return 0

def link_files(indices, source_imgs, source_lbls_list, target_img_dir, target_lbl_dir):
    """建立符號連結 (Symlink) 以節省空間"""
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for i in indices:
        src_img = source_imgs[i]
        
        # [修正 1] Image 連結：確保使用 resolve() 取得絕對路徑
        dst_img = target_img_dir / src_img.name
        if not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img)
            
        # [修正 2] Label 連結：處理 list，將找到的所有格式 (.txt, .png) 都連結過去
        labels = source_lbls_list[i]
        for src_lbl in labels:
             if src_lbl.exists():
                 dst_lbl = target_lbl_dir / src_lbl.name
                 if not dst_lbl.exists():
                     os.symlink(src_lbl.resolve(), dst_lbl)

def scan_dataset_split(dataset_path, split_name):
    """輔助函式：掃描特定資料夾並回傳資訊"""
    img_dir = dataset_path / 'images' / split_name
    lbl_dir = dataset_path / 'labels' / split_name
    
    imgs, lbls_list, grps, clss = [], [], [], []
    
    if not img_dir.exists():
        return imgs, lbls_list, grps, clss
        
    # [關鍵修改] 這裡加上 sorted()，強制依照檔名排序 (A->Z)
    # 這樣無論在哪台電腦、哪個資料夾跑，輸入順序都會固定
    files = sorted(list(img_dir.glob('*')))
    
    for f in files:
        if f.suffix.lower() not in ['.png', '.jpg', '.jpeg']: continue
        
        # [修正 3] 同時搜尋 .txt 和 .png
        found_labels = []
        l_txt = lbl_dir / f"{f.stem}.txt"
        l_png = lbl_dir / f"{f.stem}.png"
        
        if l_txt.exists(): found_labels.append(l_txt)
        if l_png.exists(): found_labels.append(l_png)
        
        # 決定類別 (優先看 png，沒有才看 txt)
        # 這是為了 Stratified 分層用的，不會影響檔案連結
        cls = 0
        target_lbl_for_class = l_png if l_png.exists() else (l_txt if l_txt.exists() else None)
        if target_lbl_for_class:
            cls = get_sample_class(target_lbl_for_class)
        
        grp = parse_scene_id(f.name)
        
        imgs.append(f)
        lbls_list.append(found_labels) # 這裡存的是 list
        grps.append(grp)
        clss.append(cls)
        
    return imgs, lbls_list, grps, clss

def prepare_kfold_datasets(original_dataset_path, results_base_dir, k_folds=3, seed=42):
    """
    執行 Stratified Group K-Fold 拆分，建立暫存資料夾，並產生 Excel 紀錄表。
    """
    # [關鍵修正] 使用 .resolve() 強制轉換為絕對路徑
    original_path = Path(original_dataset_path).expanduser().resolve()
    
    cv_work_dir = Path(results_base_dir) / "cv_temp" / original_path.name
    
    # 若資料夾已存在，先移除以確保乾淨
    if cv_work_dir.exists():
        logging.info(f"[CV] 清理舊的 CV 暫存資料夾: {cv_work_dir}")
        shutil.rmtree(cv_work_dir)
    
    # 1. 掃描 Train 與 Val (合併為開發集)
    logging.info(f"[CV] 正在掃描 Train/Val 資料集 (來源: {original_path})...")
    tr_imgs, tr_lbls, tr_grps, tr_clss = scan_dataset_split(original_path, 'train')
    val_imgs, val_lbls, val_grps, val_clss = scan_dataset_split(original_path, 'val')
    
    # 合併
    all_imgs = tr_imgs + val_imgs
    all_lbls = tr_lbls + val_lbls #這現在是 list of lists
    all_groups = tr_grps + val_grps
    all_classes = tr_clss + val_clss

    X = np.array(all_imgs)
    y = np.array(all_classes)
    groups = np.array(all_groups)
    
    # 1.1 額外掃描 Test Set (不參與拆分，但需要統計)
    logging.info("[CV] 正在掃描 Test 資料集以進行統計...")
    test_imgs, _, _, test_clss = scan_dataset_split(original_path, 'test')
    test_pos_count = sum(test_clss)
    test_neg_count = len(test_clss) - test_pos_count
    
    # 準備 Excel 紀錄 DataFrame (詳細清單)
    record_df = pd.DataFrame({
        'filename': [p.name for p in X],
        'scene_id': groups,
        'class': y, # 1=Pos, 0=Neg
        'original_path': [str(p) for p in X]
    })

    # 準備 統計摘要 列表
    summary_stats = []

    # 2. 執行拆分
    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_paths = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_name = f"fold_{fold_idx}" # 0-based
        fold_root = cv_work_dir / fold_name
        
        # 更新 Excel 詳細紀錄
        record_df[f'Fold_{fold_idx+1}_Role'] = '' # 1-based for Excel
        record_df.loc[train_idx, f'Fold_{fold_idx+1}_Role'] = 'Train'
        record_df.loc[val_idx, f'Fold_{fold_idx+1}_Role'] = 'Val'
        
        # 計算統計數據
        tr_pos = sum(y[train_idx])
        tr_neg = len(train_idx) - tr_pos
        val_pos = sum(y[val_idx])
        val_neg = len(val_idx) - val_pos
        
        logging.info(f"[CV] {fold_name}: "
                     f"Train(Pos:{tr_pos}, Neg:{tr_neg}) | "
                     f"Val(Pos:{val_pos}, Neg:{val_neg})")
        
        summary_stats.append({
            'Fold': fold_idx + 1,
            'Train_Pos': tr_pos, 'Train_Neg': tr_neg, 'Train_Total': len(train_idx),
            'Val_Pos': val_pos, 'Val_Neg': val_neg, 'Val_Total': len(val_idx),
            'Test_Pos': test_pos_count, 'Test_Neg': test_neg_count, 'Test_Total': len(test_clss)
        })
        
        # 3. 建立連結 (Train/Val)
        link_files(train_idx, X, all_lbls, fold_root / 'images' / 'train', fold_root / 'labels' / 'train')
        link_files(val_idx, X, all_lbls, fold_root / 'images' / 'val', fold_root / 'labels' / 'val')
        
        # 4. 處理 Test (Symlink 原始 Test)
        orig_test_img = original_path / 'images' / 'test'
        orig_test_lbl = original_path / 'labels' / 'test'
        if orig_test_img.exists():
            target_t_img = fold_root / 'images' / 'test'
            if not target_t_img.exists():
                 target_t_img.symlink_to(orig_test_img.resolve())
        if orig_test_lbl.exists():
            target_t_lbl = fold_root / 'labels' / 'test'
            if not target_t_lbl.exists():
                target_t_lbl.symlink_to(orig_test_lbl.resolve())
            
        fold_paths.append(str(fold_root))
    
    # 儲存 Excel (兩個 Sheet)
    excel_path = cv_work_dir / f"{original_path.name}_CV_Split_Info.xlsx"
    cv_work_dir.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Summary_Stats', index=False)
        record_df.to_excel(writer, sheet_name='Split_Details', index=False)
        
    logging.info(f"[CV] 分割詳細資訊與統計已儲存至: {excel_path}")
        
    return fold_paths