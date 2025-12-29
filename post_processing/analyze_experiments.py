import pandas as pd
from pathlib import Path
import numpy as np
import os

def analyze_experiments(results_base_path, excel_log_path, experiment_folders):
    """
    分析多組實驗，找出每組實驗中最佳的 Fold (根據 Validation IoU)，
    並從 Excel Log 中讀取該最佳 Fold 的詳細評估指標，最後計算平均值。
    """
    
    results_base = Path(results_base_path)
    excel_path = Path(excel_log_path)
    
    if not excel_path.exists():
        print(f"[Error] Excel log file not found: {excel_path}")
        return

    print(f"Loading Excel log from: {excel_path}")
    try:
        df_raw = pd.read_excel(excel_path)
        
        # 檢查是否需要轉置 (Transpose)
        # 如果第一欄包含 'Metric / Parameter'，且 'Results_folder' 在該欄的值中，則表示表格是轉置過的
        if 'Metric / Parameter' in df_raw.columns:
            # 設定第一欄為 Index
            df_raw.set_index('Metric / Parameter', inplace=True)
            # 轉置 DataFrame: 行變列，列變行
            df_excel = df_raw.T
            # 重設 Index，讓原本的 Column Names 變成一個欄位 (可選)
            # df_excel.reset_index(inplace=True)
            print("  [Info] Transposed Excel data to match expected format.")
        else:
            df_excel = df_raw

    except Exception as e:
        print(f"[Error] Failed to read Excel file: {e}")
        return

    # 定義要提取的指標
    metrics_to_collect = [
        'F1-score(pixel)',
        'Accuracy(pixel)',
        'IoU(pixel)',
        'IoU_Bg(pixel)',
        'mIoU(pixel)',
        'reconstruction_accuracy',
        'reconstruction_f1_score',
        'reconstruction_mean_iou',
        'reconstruction_iou_oil',
        'reconstruction_iou_bg'
    ]
    
    collected_data = []

    print("\n" + "="*80)
    print(f"{'Experiment':<50} | {'Best Fold':<10} | {'Val IoU':<10}")
    print("="*80)

    for exp_folder_name in experiment_folders:
        exp_path = results_base / exp_folder_name
        
        if not exp_path.exists():
            print(f"{exp_folder_name:<50} | [Not Found]")
            continue

        # 1. 找出該實驗中最佳的 Fold
        folds = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('fold_')])
        
        best_fold_name = None
        best_val_iou = -1.0
        best_fold_path = None

        for fold_dir in folds:
            log_file = fold_dir / 'training_log.csv'
            if not log_file.exists():
                continue
            
            try:
                df_log = pd.read_csv(log_file)
                if 'val_iou' in df_log.columns and not df_log.empty:
                    max_iou = df_log['val_iou'].max()
                    if max_iou > best_val_iou:
                        best_val_iou = max_iou
                        best_fold_name = fold_dir.name
                        best_fold_path = fold_dir
            except Exception:
                pass
        
        if best_fold_name:
            print(f"{exp_folder_name:<50} | {best_fold_name:<10} | {best_val_iou:.6f}")
            
            # 2. 從 Excel 中查找對應的數據
            # Excel 中的 Results_folder 通常是絕對路徑，我們需要比對
            # 為了避免路徑格式差異 (如結尾斜線)，我們統一轉為字串並正規化
            
            target_path_str = str(best_fold_path.resolve())
            
            # 在 DataFrame 中搜尋 Results_folder 欄位
            # 注意：Excel 中的路徑可能與當前系統路徑略有不同 (例如 mount point)，
            # 建議用 endswith 或 contains 來匹配較為保險，這裡嘗試精確匹配或部分匹配
            
            matched_row = None
            
            # 嘗試 1: 精確匹配
            if 'Results_folder' in df_excel.columns:
                matches = df_excel[df_excel['Results_folder'].astype(str) == target_path_str]
                if not matches.empty:
                    matched_row = matches.iloc[0]
                else:
                    # 嘗試 2: 寬鬆匹配 (只要 Excel 路徑包含 fold 資料夾名稱且包含實驗名稱)
                    # 這對於路徑有變動的情況很有用
                    mask = df_excel['Results_folder'].astype(str).apply(
                        lambda x: exp_folder_name in x and best_fold_name in x
                    )
                    matches = df_excel[mask]
                    if not matches.empty:
                        # 取最後一筆 (通常是最新的)
                        matched_row = matches.iloc[-1]
            
            if matched_row is not None:
                data_entry = {'Experiment': exp_folder_name, 'Fold': best_fold_name}
                for metric in metrics_to_collect:
                    val = matched_row.get(metric, np.nan)
                    data_entry[metric] = val
                collected_data.append(data_entry)
            else:
                print(f"   [Warning] Could not find entry in Excel for path: {target_path_str}")

        else:
            print(f"{exp_folder_name:<50} | [No Valid Log]")

    # 3. 計算統計數據並印出
    if not collected_data:
        print("\nNo data collected.")
        return

    df_results = pd.DataFrame(collected_data)
    
    print("\n" + "="*80)
    print("Collected Metrics for Best Folds:")
    print("="*80)
    print(df_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("Average Results:")
    print("="*80)
    
    # 計算平均值與標準差
    stats = []
    for metric in metrics_to_collect:
        # 確保數據是數值型
        series = pd.to_numeric(df_results[metric], errors='coerce')
        mean_val = series.mean()
        std_val = series.std()
        stats.append({
            'Metric': metric,
            'Mean': mean_val,
            'Std': std_val
        })
        print(f"{metric:<30}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    # ==========================================
    # 使用者設定區
    # ==========================================
    
    # 1. 實驗結果根目錄
    RESULTS_BASE_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/model/DV4_small_transferPaperRGB_CV"
    
    # 2. Excel Log 路徑
    EXCEL_LOG_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/excel_log/DV4_small_transferPaperRGB_CV_log.xlsx"
    
    # 3. 要分析的實驗資料夾列表
    # EXPERIMENT_FOLDERS = [
    #     "20251219-141140_Segformer_2048_transferPaperRGB",
    #     "20251219-220542_Segformer_2048_transferPaperRGB",
    #     "20251221-135401_Segformer_2048_transferPaperRGB",
    #     "20251221-194328_Segformer_2048_transferPaperRGB"
    # ]
    EXPERIMENT_FOLDERS = [
        "20251219-152043_Segformer_2048to512_transferPaperRGB_NORGB_originVV_Version",
        "20251219-231000_Segformer_2048to512_transferPaperRGB_NORGB_originVV_Version",
        "20251221-152658_Segformer_2048to512_transferPaperRGB_NORGB_originVV_Version",
        "20251221-205854_Segformer_2048to512_transferPaperRGB_NORGB_originVV_Version"
    ]
    
    
    # ==========================================
    
    analyze_experiments(RESULTS_BASE_PATH, EXCEL_LOG_PATH, EXPERIMENT_FOLDERS)
