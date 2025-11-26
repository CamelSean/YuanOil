from pathlib import Path
import yaml
import numpy as np
import cv2
from tqdm import tqdm
import torch
import shutil

from .reconstruction_module import run_reconstruction_evaluation, calculate_iou
from .training_module import get_model_adapter

def _get_gt_mask(label_path_base, h, w, architecture):
    """
    根據模型架構獲取真實標籤遮罩 (Ground Truth Mask)。
    - 對於 'yolo'，讀取 .txt 檔案並繪製多邊形。
    - 對於其他模型，讀取 .png 影像遮罩。
    """
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    has_gt_object = False

    if architecture == 'yolo':
        # YOLO 的標籤檔案是 .txt
        label_path = label_path_base.with_suffix('.txt')
        if label_path.exists() and label_path.stat().st_size > 0:
            polygons = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        has_gt_object = True
                        # 將歸一化座標轉換為絕對座標
                        poly = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                        poly[:, 0] *= w
                        poly[:, 1] *= h
                        polygons.append(poly.astype(np.int32))
            if polygons:
                # 在空白遮罩上繪製所有多邊形
                cv2.fillPoly(gt_mask, polygons, 1)
    else:
        # 其他模型 (如 DeepLabV3+) 的標籤是 .png 影像
        label_path = label_path_base.with_suffix('.png')
        if label_path.exists():
            mask_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None and mask_img.sum() > 0:
                gt_mask = (mask_img > 0).astype(np.uint8)
                has_gt_object = True
    
    return gt_mask, has_gt_object, label_path


# 檔案: main/evaluation_module.py
# (請替換這一個函式)

def generate_categorized_predictions(model_adapter, exp_config, results_path):
    """
    [v1.4 - 顏色疊加版]
    對測試集產生預測，並根據預測結果與真實標籤的比較，
    將影像分類儲存到對應的資料夾 (TP, FP, FN, TN)。
    - TP/FP 會以高透明度的 TP/FP/FN 顏色疊加儲存。
    - FN/TN 儲存原圖。
    """
    print("\n--- [Evaluation] Start: Generating categorized predictions (TP, FP, FN, TN) ---")
    try:
        architecture = exp_config.get('architecture', 'unknown')
        print(f"  - Architecture detected: '{architecture}'. Using appropriate label format.")

        dataset_cfg = exp_config.get('dataset', {})
        if not dataset_cfg:
            print("  [錯誤] 找不到 dataset 設定，無法執行分類預測。")
            return

        base_path = Path(dataset_cfg.get('path'))
        test_img_dir = base_path / dataset_cfg.get('test', 'images/test')
        
        eval_on_orig_cfg = exp_config.get('evaluation_on_original', {})
        use_original_gt = eval_on_orig_cfg.get('enabled', False)
        original_data_root = eval_on_orig_cfg.get('original_data_root')
        gt_base_path = base_path
        original_img_base_path = base_path

        if use_original_gt and original_data_root:
            print(f"  - [INFO] High-resolution evaluation ENABLED. Using ground truth from: {original_data_root}")
            gt_base_path = Path(original_data_root)
            original_img_base_path = Path(original_data_root)
        else:
            print(f"  - [INFO] High-resolution evaluation DISABLED. Using default ground truth from dataset path.")

        test_label_dir = gt_base_path / 'labels' / test_img_dir.name.replace('images/', '')
        original_test_img_dir = original_img_base_path / 'images' / test_img_dir.name.replace('images/', '')

        if not test_img_dir.is_dir() or not test_label_dir.is_dir():
            print(f"  [警告] 找不到測試圖片 ({test_img_dir}) 或標籤 ({test_label_dir}) 資料夾，跳過此任務。")
            return

        output_base_dir = results_path / "categorized_predictions"
        tp_dir = output_base_dir / "1_True_Positive"; fp_dir = output_base_dir / "2_False_Positive"
        fn_dir = output_base_dir / "3_False_Negative"; tn_dir = output_base_dir / "4_True_Negative"
        if output_base_dir.exists(): shutil.rmtree(output_base_dir)
        for d in [tp_dir, fp_dir, fn_dir, tn_dir]: d.mkdir(parents=True)
        
        print(f"  - Output directories created at: {output_base_dir}")

        # --- [FIX v1.4] 顏色和透明度設定 ---
        # BGR 格式 (與 reconstruction_module 一致)
        color_tp = (0, 255, 255)   # TP: 青色
        color_fp = (0, 0, 255)   # FP: 紅色
        color_fn = (255, 0, 0)   # FN: 藍色
        
        # 從 config 讀取透明度，預設 0.4 (您要求的 "很透明")
        alpha = exp_config.get('eval_alpha', 0.4) 
        beta = 1.0 - alpha
        print(f"  - [Info] Categorized prediction overlay alpha set to: {alpha}")
        # --- [FIX v1.4 END] ---

        image_files = list(test_img_dir.glob('*.png')) + list(test_img_dir.glob('*.jpg'))

        for img_path in tqdm(image_files, desc="Generating Categorized Predictions", mininterval=5.0):
            low_res_img = cv2.imread(str(img_path))
            if low_res_img is None: continue

            target_img_to_draw = low_res_img.copy()
            target_h, target_w = target_img_to_draw.shape[:2]

            if use_original_gt and original_data_root:
                original_img_path = next(original_test_img_dir.glob(f"{img_path.stem}.*"), None)
                if original_img_path and original_img_path.exists():
                    target_img_to_draw = cv2.imread(str(original_img_path))
                    target_h, target_w = target_img_to_draw.shape[:2]

            label_path_base = test_label_dir / img_path.stem
            gt_mask_binary, has_gt_object, _ = _get_gt_mask(label_path_base, target_h, target_w, architecture)

            results = model_adapter.predict(
                source=str(img_path),
                imgsz=exp_config.get('imgsz', 640),
                conf=exp_config.get('eval_conf', 0.25),
                boxes=False # [FIX v1.4] 強制關閉 boxes
            )

            prediction_made = results and results[0] and (
                (architecture == 'yolo' and results[0].masks is not None) or
                (architecture != 'yolo' and results[0].pred_mask_np.sum() > 0)
            )
            
            pred_mask_resized = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # --- [FIX v1.4] 統一的疊加邏輯 ---
            if prediction_made:
                if architecture == 'yolo':
                    pred_mask_low_res = torch.any(results[0].masks.data, dim=0).cpu().numpy().astype(np.uint8)
                else:
                    # 使用二值化遮罩進行分類
                    pred_mask_low_res = results[0].pred_mask_np
                
                pred_mask_resized = cv2.resize(pred_mask_low_res, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            per_image_iou = calculate_iou(pred_mask_resized, gt_mask_binary)
            output_filename = f"{img_path.stem}_iou_{per_image_iou:.4f}{img_path.suffix}"

            # 建立 TP/FP/FN 疊加影像
            temp_overlay = np.zeros_like(target_img_to_draw)
            temp_overlay[np.logical_and(pred_mask_resized, gt_mask_binary)] = color_tp
            temp_overlay[np.logical_and(pred_mask_resized, np.logical_not(gt_mask_binary))] = color_fp
            temp_overlay[np.logical_and(np.logical_not(pred_mask_resized), gt_mask_binary)] = color_fn
            
            # 將疊加層與原始影像混合
            # 只有在有東西可以顯示時 (TP, FP, 或 FN) 才進行混合
            if temp_overlay.sum() > 0:
                overlay_image = cv2.addWeighted(temp_overlay, alpha, target_img_to_draw, beta, 0)
            else:
                overlay_image = target_img_to_draw # 對於 TN，保持原圖
            
            # 根據分類儲存影像
            if prediction_made and has_gt_object:
                cv2.imwrite(str(tp_dir / output_filename), overlay_image)
            elif prediction_made and not has_gt_object:
                cv2.imwrite(str(fp_dir / output_filename), overlay_image)
            elif not prediction_made and has_gt_object:
                cv2.imwrite(str(fn_dir / output_filename), overlay_image) # FN 也顯示疊加 (顯示藍色的 GT)
            else: 
                # TN (True Negative)
                shutil.copy(img_path, tn_dir / f"{img_path.stem}_iou_1.0000{img_path.suffix}")
            # --- [FIX v1.4 END] ---

        print(f"  - Categorized predictions saved successfully.")
        print("--- [Evaluation] End: Generating categorized predictions ---")

    except Exception as e:
        print(f"  [錯誤] 產生分類化預測圖時發生錯誤: {e}")
        import traceback; traceback.print_exc()

def calculate_pixel_level_metrics(model_adapter, exp_config):
    """
    計算 patch 層級的像素指標，例如像素準確率 (Pixel Accuracy) 和 IoU。
    [v1.5] 新增 Background IoU 和 Mean IoU (mIoU) 的計算。
    """
    print("\n--- [Evaluation] Start: Calculating pixel-level metrics (Accuracy, IoU, mIoU, F1-score) ---")
    try:
        architecture = exp_config.get('architecture', 'unknown')
        print(f"  - Architecture detected: '{architecture}'. Using appropriate label format.")

        dataset_cfg = exp_config.get('dataset', {});
        if not dataset_cfg: return {}
        base_path = Path(dataset_cfg['path']); 
        test_img_dir = base_path / dataset_cfg.get('test', 'images/test')

        eval_on_orig_cfg = exp_config.get('evaluation_on_original', {})
        use_original_gt = eval_on_orig_cfg.get('enabled', False)
        original_data_root = eval_on_orig_cfg.get('original_data_root')
        gt_base_path = base_path
        original_img_base_path = base_path
        
        if use_original_gt and original_data_root:
            print(f"  - [INFO] High-resolution evaluation ENABLED. Using ground truth from: {original_data_root}")
            gt_base_path = Path(original_data_root)
            original_img_base_path = Path(original_data_root)
        else:
            print(f"  - [INFO] High-resolution evaluation DISABLED. Using default ground truth from dataset path.")

        test_label_dir = gt_base_path / 'labels' / test_img_dir.name.replace('images/', '')
        original_test_img_dir = original_img_base_path / 'images' / test_img_dir.name.replace('images/', '')
        if not test_label_dir.is_dir(): return {}
        
        image_files = list(test_img_dir.glob('*.png')) + list(test_img_dir.glob('*.jpg'))
        if not image_files: return {}
        
        total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

        for img_path in tqdm(image_files, desc="Calculating Pixel-Level Metrics", mininterval=5.0):
            low_res_img = cv2.imread(str(img_path))
            if low_res_img is None: continue
            target_h, target_w = low_res_img.shape[:2]
            
            if use_original_gt and original_data_root:
                original_img_path = next(original_test_img_dir.glob(f"{img_path.stem}.*"), None)
                if original_img_path and original_img_path.exists():
                    target_h, target_w = cv2.imread(str(original_img_path)).shape[:2]
            
            label_path_base = test_label_dir / img_path.stem
            gt_mask_binary, _, _ = _get_gt_mask(label_path_base, target_h, target_w, architecture)

            results = model_adapter.predict(
                source=str(img_path),
                imgsz=exp_config.get('imgsz', 640),
                conf=exp_config.get('eval_conf', 0.25),
                # boxes=False # 強制關閉 boxes，與 reconstruction 一致
            )
            
            pred_mask_resized = np.zeros((target_h, target_w), dtype=np.uint8)
            if results and results[0]:
                # 這裡我們只需要二值化遮罩來計算指標
                if architecture == 'yolo':
                    if results[0].masks is not None:
                        pred_mask_low_res = torch.any(results[0].masks.data, dim=0).cpu().numpy().astype(np.uint8)
                    else:
                        pred_mask_low_res = np.zeros((low_res_img.shape[0], low_res_img.shape[1]), dtype=np.uint8)
                else:
                    # 對於非 YOLO 模型，直接使用 pred_mask_binary_np
                    pred_mask_low_res = results[0].pred_mask_binary_np
                
                if pred_mask_low_res.shape[:2] != (target_h, target_w):
                     pred_mask_resized = cv2.resize(pred_mask_low_res, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                else:
                     pred_mask_resized = pred_mask_low_res
            
            total_tp += np.sum(np.logical_and(pred_mask_resized, gt_mask_binary))
            total_tn += np.sum(np.logical_and(np.logical_not(pred_mask_resized), np.logical_not(gt_mask_binary)))
            total_fp += np.sum(np.logical_and(pred_mask_resized, np.logical_not(gt_mask_binary)))
            total_fn += np.sum(np.logical_and(np.logical_not(pred_mask_resized), gt_mask_binary))
        
        epsilon = 1e-9 
        
        pixel_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
        pixel_iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
        pixel_precision = total_tp / (total_tp + total_fp + epsilon)
        pixel_recall = total_tp / (total_tp + total_fn + epsilon)
        pixel_f1_score = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall + epsilon)

        # --- [FIX v1.5] 新增 Background IoU 和 Mean IoU ---
        pixel_iou_bg = total_tn / (total_tn + total_fn + total_fp + epsilon)
        pixel_mean_iou = (pixel_iou + pixel_iou_bg) / 2
        # --- [FIX v1.5 END] ---

        print(f"  - Pixel-level Accuracy: {pixel_accuracy:.4f}")
        print(f"  - Pixel-level IoU (Oil): {pixel_iou:.4f}")
        print(f"  - Pixel-level IoU (Bg): {pixel_iou_bg:.4f}")
        print(f"  - Pixel-level mIoU:     {pixel_mean_iou:.4f}")
        print(f"  - Pixel-level Precision: {pixel_precision:.4f}")
        print(f"  - Pixel-level Recall: {pixel_recall:.4f}")
        print(f"  - Pixel-level F1-score: {pixel_f1_score:.4f}")
        print("--- [Evaluation] End: Calculating pixel-level metrics ---")
        
        return {
            'Accuracy(pixel)': f"{pixel_accuracy:.4f}", 
            'IoU(pixel)': f"{pixel_iou:.4f}",
            'IoU_Bg(pixel)': f"{pixel_iou_bg:.4f}",     # 新增
            'mIoU(pixel)': f"{pixel_mean_iou:.4f}",      # 新增
            'Precision(pixel)': f"{pixel_precision:.4f}",
            'Recall(pixel)': f"{pixel_recall:.4f}",
            'F1-score(pixel)': f"{pixel_f1_score:.4f}"
        }
    except Exception as e:
        print(f"  [錯誤] 計算像素級指標時出錯: {e}"); import traceback; traceback.print_exc()
        return {}

def evaluate_and_visualize(exp_config, data_yaml_path, model_path, results_path):
    exp_name = exp_config.get('test_name') or exp_config['experiment_name']
    print(f"\n--- [Evaluation] Start: Full evaluation for experiment '{exp_name}' ---")
    try:
        adapter_config = {
            'architecture': exp_config.get('architecture'),
            'base_model': model_path,
            'architecture_cfg': exp_config.get('architecture_cfg', {}),
            'dataset': exp_config.get('dataset', {})
        }
        model_adapter = get_model_adapter(adapter_config)
            
        eval_params = {
            'data': str(data_yaml_path),
            'split': 'test',
            'imgsz': exp_config.get('imgsz', 640),
            'conf': exp_config.get('eval_conf', 0.25),
            'iou': exp_config.get('eval_iou', 0.6)
        }
        # 為了與 YOLOv8 的 .val() 函式相容，需傳入 project 和 name
        eval_params['project'] = str(results_path)
        eval_params['name'] = "standard_evaluation_charts"
        eval_params['exist_ok'] = True
        
        metrics = model_adapter.val(**eval_params)
        
        eval_results = {}
        if hasattr(metrics, 'box') and metrics.box.map is not None:
            p,r = metrics.box.mp, metrics.box.mr; eval_results.update({'Precision(B)':f"{p:.4f}", 'Recall(B)':f"{r:.4f}", 'mAP50(B)':f"{metrics.box.map50:.4f}", 'mAP50-95(B)':f"{metrics.box.map:.4f}", 'F1-score(B)':f"{2*p*r/(p+r+1e-9):.4f}"})
        if hasattr(metrics, 'seg') and metrics.seg.map is not None:
            p_seg,r_seg=metrics.seg.mp,metrics.seg.mr; eval_results.update({'Precision(M)':f"{p_seg:.4f}", 'Recall(M)':f"{r_seg:.4f}", 'mAP50(M)':f"{metrics.seg.map50:.4f}", 'mAP50-95(M)':f"{metrics.seg.map:.4f}", 'F1-score(M)':f"{2*p_seg*r_seg/(p_seg*r_seg+1e-9):.4f}"})

        pixel_metrics = calculate_pixel_level_metrics(model_adapter, exp_config)
        eval_results.update(pixel_metrics)

        generate_categorized_predictions(model_adapter, exp_config, results_path)

        recon_config = exp_config.get('reconstruction')
        if recon_config and recon_config.get('enabled'):
            # --- [修改] 呼叫重建 (Reconstruction) 功能 ---
            print(f"\n--- [Evaluation] Start: Reconstruction evaluation for '{exp_name}' ---")
            
            # 獲取 patch (裁切圖) 的測試路徑
            dataset_cfg = exp_config.get('dataset', {});
            base_path = Path(dataset_cfg['path']); 
            test_img_dir = base_path / dataset_cfg.get('test', 'images/test')
            
            # 獲取原始大圖的路徑
            original_data_root = Path(recon_config.get('original_data_root'))
            
            train_cfg = exp_config.get('train', {})
            imgsz_to_use = exp_config.get('imgsz', train_cfg.get('imgsz', 640))

            # 準備視覺化參數 (從 exp_config 讀取，提供預設值)
            vis_params = {
                'min_conf': exp_config.get('eval_conf', 0.25),
                'nms_iou': exp_config.get('eval_iou', 0.6),
                'alpha': recon_config.get('alpha', 0.5), # 可在 yaml 的 reconstruction 下設定 alpha
                
                # --- [FIX 2a] ---
                # 新增：將 'original_patch_size' 從 reconstruction_config 傳遞到 vis_params
                # 如果未在 yaml 中提供，預設為 'imgsz' (保持舊有行為)
                'original_patch_size': recon_config.get('original_patch_size', imgsz_to_use)
                # --- [FIX 2a END] ---
            }

            recon_metrics = run_reconstruction_evaluation(
                model_adapter=model_adapter,
                test_image_dir=test_img_dir,
                original_data_root=original_data_root,
                results_path=results_path,
                imgsz=imgsz_to_use,
                vis_params=vis_params
            )
            eval_results.update(recon_metrics)
            print(f"--- [Evaluation] End: Reconstruction evaluation for '{exp_name}' ---")
            # --- [修改] 結束 ---
        
        print(f"--- [Evaluation] End: Full evaluation for '{exp_name}' ---")
        return eval_results
    except Exception as e:
        print(f"[Error] An error occurred during evaluation: {e}"); import traceback; traceback.print_exc()
        return {"error": str(e)}