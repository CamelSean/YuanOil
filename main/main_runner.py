# 專案主進入點
import yaml
from pathlib import Path
import datetime
import copy
import torch
import gc
import os

# 從本地模組導入
from .training_module import train_model
from .evaluation_module import evaluate_and_visualize, run_advanced_evaluation
from .tracking_module import log_to_excel
from .reconstruction_module import run_reconstruction_pipeline
from .training_module import get_model_adapter
from .utils import (
    get_image_counts, create_temp_data_yaml,
)
# 導入 CV 工具
from .cross_validation_utils import prepare_kfold_datasets

# 取得此腳本所在的目錄
script_dir = Path(__file__).parent


def run_evaluation_job(exp_config, model_path, results_path, excel_path, desired_order, run_timestamp, training_metrics=None):
    """
    執行單個評估任務，包括模型預測、視覺化和結果記錄。
    """
    test_name = exp_config.get('test_name', exp_config.get('experiment_name'))
    print(f"\n--- 開始評估任務: {test_name} ---")
    print(f"--- 使用模型: {model_path} ---")

    # 準備評估用的資料集設定
    eval_dataset_config = exp_config.get('dataset', {})
    if not eval_dataset_config:
        print("   [錯誤] 評估任務未定義 'dataset'。")
        return

    # 為資料集設定提供預設值
    for split in ['train', 'val', 'test']:
        eval_dataset_config.setdefault(split, f'images/{split}')
    eval_dataset_config.setdefault('nc', 1)
    eval_dataset_config.setdefault('names', ['oil'])
    
    # 建立評估結果目錄和臨時資料設定檔
    eval_results_path = results_path
    eval_results_path.mkdir(exist_ok=True, parents=True)
    temp_yaml_path = create_temp_data_yaml(eval_dataset_config, eval_results_path)

    # 準備要記錄到 Excel 的日誌資料
    log_data = {k: v for k, v in exp_config.items() if not isinstance(v, (dict, list))}
    log_data['run_timestamp'] = run_timestamp
    log_data['Experiment_name'] = log_data.pop('experiment_name', 'N/A')
    if 'test_name' in log_data:
        log_data['Test name'] = log_data.pop('test_name')

    # 格式化日誌欄位名稱
    train_params = exp_config.get('train', {})
    log_data['Epochs'] = train_params.get('epochs')
    log_data['Batch size'] = train_params.get('batch_size')
    log_data['Image size'] = train_params.get('imgsz')
    log_data['Patience'] = train_params.get('patience')

    # 載入額外的超參數
    log_data['mosaic'] = train_params.get('mosaic')
    log_data['degrees'] = train_params.get('degrees')
    log_data['translate'] = train_params.get('translate')
    log_data['scale'] = train_params.get('scale')
    log_data['dropout'] = train_params.get('dropout')

    # 合併訓練階段的指標
    if training_metrics:
        log_data.update(training_metrics)

    # 獲取圖片數量並更新日誌
    log_data.update(get_image_counts(eval_dataset_config, eval_results_path))
    log_data['best_model_path'] = str(model_path)
    log_data['Results_folder'] = str(eval_results_path)

    # 執行評估與視覺化
    eval_metrics = evaluate_and_visualize(exp_config, temp_yaml_path, model_path, eval_results_path)
    if eval_metrics:
        log_data.update(eval_metrics)

    # ========================================================
    # 準備模型 Adapter (供後續進階分析與重建使用)
    # ========================================================
    adapter_for_eval = None
    try:
        # 建立 Config，強制指定 'base_model' 為當前評估的模型路徑
        config_for_eval = exp_config.copy()
        config_for_eval['base_model'] = str(model_path) 
        
        # 初始化 Adapter
        adapter_for_eval = get_model_adapter(config_for_eval)
        
        # 取得 imgsz (預設 512)
        imgsz = exp_config.get('train', {}).get('imgsz', 512)

        gradcam_config = exp_config.get('grad_cam', {'enabled': False})
        
    except Exception as e:
        print(f"[Warning] 無法載入模型 Adapter，後續進階分析將跳過: {e}")
        import traceback; traceback.print_exc()

    if adapter_for_eval:
        # --- [2] 呼叫 Patch-level 進階評估 (HD95, Missed, FP) ---
        print("\n>>> 步驟 2/3: 執行 Patch-level 進階分析 (HD95, Missed)...")
        try:
            run_advanced_evaluation(
                model_adapter=adapter_for_eval, 
                dataset_config=exp_config['dataset'], 
                save_dir=results_path,
                imgsz=imgsz,
                gradcam_config=gradcam_config
            )
        except Exception as e:
            print(f"[Warning] Patch-level 進階評估執行失敗: {e}")
            import traceback; traceback.print_exc()

        # --- [3] 呼叫 Reconstruction 進階評估 (若有開啟) ---
        # 檢查 config 中是否有開啟 reconstruction (通常在 post_tests 裡設定)
        if exp_config.get('reconstruction', False):
            print("\n>>> 步驟 3/3: 執行大圖重建與進階分析...")
            recon_cfg = exp_config.get('reconstruction')
            if isinstance(recon_cfg, bool): recon_cfg = {}
            explicit_root = None
            if 'original_data_root' in recon_cfg:
                explicit_root = Path(recon_cfg['original_data_root'])
            elif 'dataset' in exp_config and 'path' in exp_config['dataset']:
                 # 備用：有時候 original root 跟 dataset root 有關
                 pass 

            try:
                run_reconstruction_pipeline(
                    model_adapter=adapter_for_eval,
                    dataset_config=exp_config['dataset'],
                    save_dir=results_path,
                    gradcam_config=gradcam_config,
                    reconstruction_config=recon_cfg,
                    explicit_original_root=explicit_root  
                )
            except Exception as e: print(f"[Err] Recon: {e}")
        else:
            print("\n>>> 步驟 3/3: 跳過大圖重建 (reconstruction: False)")
    # ========================================================

    # 根據預設順序排序日誌並寫入 Excel
    ordered_log_data = {key: log_data.get(key) for key in desired_order}
    ordered_log_data.update({k: v for k, v in log_data.items() if k not in desired_order})
    
    log_to_excel(excel_path, ordered_log_data, desired_order)


def run_experiment_core(current_exp_config, results_path, excel_path, desired_order, timestamp, completed_experiments_paths):
    """
    封裝核心的實驗流程：訓練 -> 儲存 -> 後處理測試。
    這樣做是為了支援 CV 迴圈可以重複呼叫此邏輯。
    """
    model_to_evaluate = None
    training_metrics = {}

    # --- 訓練或測試模式 ---
    if current_exp_config['mode'] == 'train':
        training_results = train_model(current_exp_config, results_path)
        if training_results and 'best_model_path' in training_results:
            model_to_evaluate = training_results['best_model_path']
            training_metrics = training_results
        else:
            print(f"[錯誤] 實驗 '{current_exp_config['experiment_name']}' 訓練失敗，跳過後續步驟。")
            completed_experiments_paths[current_exp_config['experiment_name']] = None
            return # 訓練失敗直接返回
    
    elif current_exp_config['mode'] == 'test':
        model_to_evaluate = current_exp_config['base_model']
        training_metrics = {} 
        if not model_to_evaluate or not Path(model_to_evaluate).exists():
            print(f"[錯誤] 測試模式下找不到模型: {model_to_evaluate}")
            return
    
    # --- 後處理測試 ---
    if model_to_evaluate:
        post_tests = current_exp_config.get('post_tests', [])
        if not post_tests:
            # 如果沒有定義 post_tests，則對主實驗進行一次評估
            print(f"\n沒有 post_tests, 對主任務 '{current_exp_config['experiment_name']}' 進行評估。")
            run_evaluation_job(current_exp_config, model_to_evaluate, results_path, excel_path, desired_order, timestamp, training_metrics)
        else:
            print(f"\n偵測到 post_tests, 開始執行後續測試任務...")
            for test_job in post_tests:
                if not test_job.get('run', True):
                    continue

                test_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                test_results_path = results_path / f"post_test_{test_timestamp}_{test_job['test_name']}"

                # 組合設定：測試任務的設定會覆寫主實驗的設定
                test_run_config = {**current_exp_config, **test_job}
                
                # 自動更新 CV 模式下的資料集路徑 (如果 test 使用的是原訓練集)
                # 這是為了確保 fold 的測試是在該 fold 的 split 上進行
                original_dataset_path = current_exp_config.get('original_dataset_path_for_cv', None)
                if original_dataset_path and test_run_config.get('dataset', {}).get('path') == original_dataset_path:
                     test_run_config['dataset']['path'] = current_exp_config['dataset']['path']
                     print(f"  [CV] Post-test '{test_job['test_name']}' 自動指向 Fold 資料集")

                run_evaluation_job(test_run_config, model_to_evaluate, test_results_path, excel_path, desired_order, test_timestamp, training_metrics)

    # 記錄成功訓練的實驗結果路徑
    if current_exp_config['mode'] == 'train':
        completed_experiments_paths[current_exp_config['experiment_name']] = model_to_evaluate


def main():
    """
    專案主進入點函式，負責讀取設定、管理實驗流程。
    """
    try:
        config_path = script_dir / 'experiments.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            master_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到設定檔 '{config_path}'！請確認檔案是否存在。")
        return

    results_base_dir = Path(master_config['results_base_dir'])
    excel_path = master_config['excel_log_path']
    completed_experiments_paths = {}

    # 定義 Excel 欄位的理想順序
    desired_order = [
        'Experiment_name', 'Test name', 'run_timestamp', 'training_time_minutes',
        'Ram_GB', 'gpu_name', 'mode', 'architecture', 'Results_folder', 'best_model_path',
        'base_model', 'Epochs', 'Batch size', 'Image size', 'Patience', 
        'mosaic', 'degrees', 'translate', 'scale', 'dropout',
        'train_count', 'val_count', 'test_count',
        'Precision(B)', 'Recall(B)', 'mAP50(B)', 'mAP50-95(B)', 'F1-score(B)',
        'Precision(M)', 'Recall(M)', 'mAP50(M)', 'mAP50-95(M)', 'F1-score(M)',
        'Precision(pixel)', 'Recall(pixel)', 'F1-score(pixel)','Accuracy(pixel)', 
        'IoU(pixel)', 'IoU_Bg(pixel)', 'mIoU(pixel)', 
        'reconstruction_accuracy', 'reconstruction_f1_score', 'reconstruction_mean_iou',
        'reconstruction_iou_oil', 'reconstruction_iou_bg'
    ]

    for exp_config in master_config.get('experiments', []):
        
        try:
            if not exp_config.get('run', True):
                print(f"\n--- 跳過實驗: {exp_config.get('experiment_name')} (run 設為 False) ---")
                continue

            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            
            # 建立單一的主實驗資料夾
            # 例如: .../20231219-100000_ExpName/
            base_results_path = results_base_dir / f"{timestamp}_{exp_config['experiment_name']}"
            base_results_path.mkdir(exist_ok=True, parents=True)

            print("\n" + "="*80)
            print(f"==> 開始處理實驗: {exp_config.get('experiment_name')}")
            print(f"==> 結果路徑: {base_results_path}")

            # 檢查是否啟用 Cross Validation
            cv_cfg = exp_config.get('cross_validation', {})
            cv_enabled = (exp_config.get('mode') == 'train') and cv_cfg.get('enabled', False)

            if cv_enabled:
                print(f"==> [CV模式] 啟用 Cross Validation (K={cv_cfg.get('k_folds', 3)})...")
                k_folds = cv_cfg.get('k_folds', 3)
                seed = cv_cfg.get('random_state', 42)
                
                # 準備資料集
                fold_paths = prepare_kfold_datasets(
                    original_dataset_path=exp_config['dataset']['path'],
                    results_base_dir=results_base_dir,
                    k_folds=k_folds,
                    seed=seed
                )

                # 迴圈執行每一折
                for i, fold_dataset_path in enumerate(fold_paths):
                    fold_name = f"fold_{i+1}"
                    print(f"\n  >>> 執行 {fold_name} / {k_folds} <<<")
                    
                    # 建立子資料夾: .../ExpName/fold_1/
                    fold_results_dir = base_results_path / fold_name
                    fold_results_dir.mkdir(exist_ok=True)
                    
                    # 準備 Fold 的設定
                    fold_config = copy.deepcopy(exp_config)
                    # 修改名稱以利 Excel 區分 (ExpName_fold_1)
                    fold_config['experiment_name'] = f"{exp_config['experiment_name']}_{fold_name}"
                    # 記錄原始路徑以便 Post-test 比對
                    fold_config['original_dataset_path_for_cv'] = exp_config['dataset']['path']
                    # 指向 Fold 資料集
                    fold_config['dataset']['path'] = fold_dataset_path
                    # 關閉子實驗的 CV 設定
                    fold_config['cross_validation']['enabled'] = False

                    config_save_path = fold_results_dir / 'exp_config.yaml'
                    print(f"[Info] 備份實驗設定至: {config_save_path}")
                    with open(config_save_path, 'w', encoding='utf-8') as f:
                        yaml.dump(fold_config, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
                    
                    # 執行核心流程 (訓練 -> 測試 -> 記錄)
                    run_experiment_core(
                        fold_config, 
                        fold_results_dir, 
                        excel_path, 
                        desired_order, 
                        timestamp, 
                        completed_experiments_paths
                    )
            
            else:
                config_save_path = base_results_path / 'exp_config.yaml'
                print(f"[Info] 備份實驗設定至: {config_save_path}")
                with open(config_save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(exp_config, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

                # 一般模式 (非 CV)
                run_experiment_core(
                    copy.deepcopy(exp_config), 
                    base_results_path, 
                    excel_path, 
                    desired_order, 
                    timestamp, 
                    completed_experiments_paths
                )

        except Exception as e:
            print(f"[嚴重錯誤] 實驗 '{exp_config.get('experiment_name')}' 執行期間發生未預期的錯誤: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print(f"\n--- 完成實驗區塊，正在進行記憶體清理... ---")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n所有已設定的實驗執行完畢！")


if __name__ == '__main__':
    # 為了讓動態導入 `adapters` 正常工作，需要從專案的根目錄來執行
    # 例如: `export PYTHONPATH=$PYTHONPATH:/path/to/your/OIL_PROJECT`
    # 然後執行: `python -m OIL_PROJECT.code_10_7.main.main_runner`
    # export PYTHONPATH=$PYTHONPATH:$(pwd) && nohup /home/sean/.conda/envs/YuansCode/bin/python -u -m main.main_runner > runner.log 2>&1 &
    main()