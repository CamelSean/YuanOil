import pandas as pd
from pathlib import Path
import numpy as np

def analyze_best_fold_reconstruction(results_base_path, experiment_folders):
    """
    1. 遍歷指定的實驗資料夾。
    2. 在每個實驗中，讀取各 Fold 的 training_log.csv，找出 Val IoU 最高的 Fold (Best Fold)。
    3. 讀取 Best Fold 下的 'Reconstruction_Advanced_Analysis.xlsx'。
    4. 計算該 Fold 的統計數據：
       - HD95: 平均值 (Mean)
       - GT Count: 總和 (Sum) -> 代表該測試集總共有多少物件
       - Missed: 總和 (Sum)
       - False Alarm: 總和 (Sum)
    """
    results_base = Path(results_base_path)
    
    print("\n" + "="*100)
    print(f"{'Experiment':<50} | {'Best Fold':<10} | {'Val IoU':<8} | {'Avg HD95':<10} | {'Total GT':<8} | {'Missed':<8} | {'False Alarm':<11}")
    print("="*100)

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

        # --- 2. 讀取 Best Fold 的 Reconstruction Analysis ---
        # 優先尋找根目錄下的檔案，若無則尋找 post_test 子目錄下的
        analysis_file = best_fold_path / 'Reconstruction_Advanced_Analysis.xlsx'
        
        if not analysis_file.exists():
            # 嘗試搜尋 post_test 資料夾
            post_test_files = list(best_fold_path.glob('post_test_*/Advanced_Analysis.xlsx'))
            if post_test_files:
                # 取最新的或是第一個
                analysis_file = post_test_files[0]
            else:
                print(f"{exp_folder:<50} | {best_fold_name:<10} | {best_val_iou:.4f}   | [No Analysis File]")
                continue

        try:
            df_analysis = pd.read_excel(analysis_file)
            
            # --- 3. 計算統計數據 ---
            # HD95 取平均 (代表整體形狀吻合度)
            avg_hd95 = df_analysis['HD95'].mean()
            
            # GT, Missed, False Alarm 取總和 (代表整個測試集的數量)
            total_gt = df_analysis['GT_Count'].sum()
            total_missed = df_analysis['Missed'].sum()
            total_fa = df_analysis['False_Alarm'].sum()
            
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

    # --- 4. 計算所有實驗的平均表現 (Optional) ---
    if collected_metrics:
        df_metrics = pd.DataFrame(collected_metrics)
        print("-" * 100)
        print("Average across these experiments:")
        print(f"Mean HD95:        {df_metrics['Avg HD95'].mean():.4f}")
        print(f"Mean Missed:      {df_metrics['Total Missed'].mean():.2f}")
        print(f"Mean False Alarm: {df_metrics['Total False Alarm'].mean():.2f}")
        print(f"(GT Count should be constant: {df_metrics['Total GT'].mean():.1f})")

if __name__ == "__main__":
    # ==========================================
    # 使用者設定區
    # ==========================================
    
    RESULTS_BASE_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/model/DV4_small_transferPaperRGB_CV"
    
    EXPERIMENT_FOLDERS = [
        "20251219-141140_Segformer_2048_transferPaperRGB",
        "20251219-220542_Segformer_2048_transferPaperRGB",
        "20251221-135401_Segformer_2048_transferPaperRGB",
        "20251221-194328_Segformer_2048_transferPaperRGB"
    ]
    
    # ==========================================
    
    analyze_best_fold_reconstruction(RESULTS_BASE_PATH, EXPERIMENT_FOLDERS)
