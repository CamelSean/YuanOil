import pandas as pd
from pathlib import Path
import numpy as np

def analyze_patch_level_advanced(results_base_path, experiment_folders):
    """
    分析 Patch-level 的 Advanced_Analysis.xlsx。
    1. 遍歷實驗與 Fold，找出最佳 Fold (根據 training_log.csv 的 val_iou)。
    2. 讀取該 Best Fold 下 post_test 資料夾中的 'Advanced_Analysis.xlsx'。
    3. 讀取該 Excel 的最後一列 (Summary Row)，提取統計資訊。
    4. 彙整並印出報告。
    """
    results_base = Path(results_base_path)
    
    print("\n" + "="*110)
    print(f"{'Experiment':<50} | {'Best Fold':<10} | {'Val IoU':<8} | {'Avg HD95':<10} | {'Total GT':<8} | {'Missed':<8} | {'False Alarm':<11}")
    print("="*110)

    collected_metrics = []

    for exp_folder in experiment_folders:
        exp_path = results_base / exp_folder
        if not exp_path.exists():
            print(f"{exp_folder:<50} | [Not Found]")
            continue

        # --- 1. 找出最佳 Fold ---
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
        
        if not best_fold_name:
            print(f"{exp_folder:<50} | [No Log]")
            continue

        # --- 2. 尋找 Patch-level Advanced Analysis ---
        # 通常位於 post_test_.../Advanced_Analysis.xlsx
        # 我們需要找到對應的 post_test 資料夾
        
        target_file = None
        post_test_dirs = sorted(list(best_fold_path.glob('post_test_*')))
        
        # 如果有多個 post_test，通常取最新的，或者根據名稱篩選
        # 這裡假設取最新的
        if post_test_dirs:
            # 排序以確保取到最新的 (如果檔名包含時間戳)
            latest_post_test = post_test_dirs[-1]
            target_file = latest_post_test / 'Advanced_Analysis.xlsx'
        
        if not target_file or not target_file.exists():
             print(f"{exp_folder:<50} | {best_fold_name:<10} | {best_val_iou:.4f}   | [No Patch Analysis File]")
             continue

        try:
            df_analysis = pd.read_excel(target_file)
            
            # --- 3. 讀取 Summary Row (最後一列) ---
            # 檢查最後一列是否為 Summary (通常 filename 會標記為 'Average' 或 'Total')
            last_row = df_analysis.iloc[-1]
            
            # 根據您的描述，最後一列已經有初步統計
            # HD95 通常是平均值
            # GT_Count, Missed, False_Alarm 通常是總和
            
            avg_hd95 = last_row['HD95']
            total_gt = last_row['GT_Count']
            total_missed = last_row['Missed']
            total_fa = last_row['False_Alarm']
            
            print(f"{exp_folder:<50} | {best_fold_name:<10} | {best_val_iou:.4f}   | {avg_hd95:.4f}     | {int(total_gt):<8} | {int(total_missed):<8} | {int(total_fa):<11}")
            
            collected_metrics.append({
                'Experiment': exp_folder,
                'Avg HD95': avg_hd95,
                'Total GT': total_gt,
                'Total Missed': total_missed,
                'Total False Alarm': total_fa
            })

        except Exception as e:
            print(f"{exp_folder:<50} | {best_fold_name:<10} | {best_val_iou:.4f}   | [Error: {e}]")

    # --- 4. 計算所有實驗的平均表現 ---
    if collected_metrics:
        df_metrics = pd.DataFrame(collected_metrics)
        print("-" * 110)
        print("Average across these experiments (Patch Level):")
        print(f"Mean HD95:        {df_metrics['Avg HD95'].mean():.4f}")
        print(f"Mean Missed:      {df_metrics['Total Missed'].mean():.2f}")
        print(f"Mean False Alarm: {df_metrics['Total False Alarm'].mean():.2f}")
        print(f"(GT Count should be constant: {df_metrics['Total GT'].mean():.1f})")

if __name__ == "__main__":
    # ==========================================
    # 使用者設定區
    # ==========================================
    
    RESULTS_BASE_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/model/DV4_small_transferPaperRGB_CV"
    
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
    
    analyze_patch_level_advanced(RESULTS_BASE_PATH, EXPERIMENT_FOLDERS)
