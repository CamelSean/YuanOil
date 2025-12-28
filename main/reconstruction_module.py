# ===================================================================
# ===           main/reconstruction_module.py (v1.1)            ===
# ===  (已修正 Bug: 不應對 adapter 傳來的 mask 重複使用 sigmoid)  ===
# ===================================================================

import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm
from torchvision.ops import nms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def stitch_masks(patch_predictions_with_scores, original_size, architecture):
    """
    【v1.2 - 機率感知版】將來自多個圖塊(patch)的分割遮罩，透過「加權平均」的方式平滑地拼接到一張原始大圖上。
    
    [FIX v1.2] 
    - 根據架構決定如何處理傳入的 mask_tensor:
    - 如果是 'yolo', mask_tensor 是 logits (原始輸出)，需要 sigmoid。
    - 如果是 'unet', 'segformer' 等, mask_tensor 是 sigmoid-ed probabilities (0.0~1.0)。
    """
    stitched_canvas = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
    weight_map = np.zeros_like(stitched_canvas, dtype=np.float32)
    
    canvas_h, canvas_w = stitched_canvas.shape

    for mask_tensor, score, x, y in patch_predictions_with_scores:
        if mask_tensor is not None:
            
            # --- [FIX v1.2] 根據架構決定如何處理 mask ---
            if architecture == 'yolo':
                 # YOLO 傳遞的是 logits tensor，需要 sigmoid
                 mask_np = torch.sigmoid(mask_tensor).cpu().numpy().astype(np.float32)
            else:
                 # 其他 adapter 傳遞的是 0.0~1.0 的機率 tensor，直接轉換
                 mask_np = mask_tensor.cpu().numpy().astype(np.float32)
            # --- [FIX v1.2 END] ---
            
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(0)
            
            h, w = mask_np.shape
            # 邊界檢查
            h_valid = min(h, canvas_h - y)
            w_valid = min(w, canvas_w - x)
            if h_valid <= 0 or w_valid <= 0:
                continue

            mask_valid = mask_np[0:h_valid, 0:w_valid]
            stitched_canvas[y : y + h_valid, x : x + w_valid] += mask_valid * score
            weight_map[y : y + h_valid, x : x + w_valid] += score
            
    safe_weight_map = weight_map.copy()
    safe_weight_map[safe_weight_map == 0] = 1
    final_mask_prob = stitched_canvas / safe_weight_map
    
    # 最終決定：在所有機率都平均完畢後，才進行 0.5 閾值判斷
    return (final_mask_prob > 0.5).astype(np.uint8)


def calculate_iou(mask1, mask2):
    """計算兩個二值化遮罩的 IoU。"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-9)

def draw_segmentation_contours(image, mask, color=(0, 255, 255), thickness=2):
    """
    在影像上繪製分割遮罩的輪廓線。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)
    return image

def draw_final_boxes(image, boxes, scores, labels):
    """在影像上繪製最終經過 NMS 處理的邊界框。"""
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        score = scores[i]
        label = labels[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        text = f"{label} {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 255), -1)
        cv2.putText(image, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image

def run_reconstruction_evaluation(model_adapter, test_image_dir, original_data_root, results_path, imgsz, vis_params):
    """
    執行 Patch 重組評估。
    [FIX v1.4] 
    - 傳遞 architecture 給 stitch_masks
    - 優先從 results[0] 獲取 'pred_mask_prob_np' (機率圖)
    - 對 YOLO 則獲取 'masks.data' (logits)
    - [您的要求] 在 predict 參數中加入 'boxes=False'
    - [您的要求] 移除了 NMS、draw_final_boxes 和 draw_segmentation_contours 邏輯
    """
    print("\n--- [Info] Start: Running Patch Reconstruction Evaluation ---")
    
    architecture = model_adapter.config.get('architecture', 'unknown')
    print(f"  - [Info] Reconstruction running for architecture: '{architecture}'")
    
    original_images_dir = original_data_root / 'images' / 'test'
    original_gt_dir = original_data_root / 'labels' / 'test'
    if not original_images_dir.is_dir() or not original_gt_dir.is_dir(): 
        print(f"  [ERROR] Reconstruction failed: Cannot find original images or labels at:")
        print(f"  - {original_images_dir}")
        print(f"  - {original_gt_dir}")
        print(f"  - 請檢查 'reconstruction.original_data_root' 路徑是否正確指向「原始大圖」資料集。")
        return {}

    all_patch_files = list(test_image_dir.glob('*.jpg')) + list(test_image_dir.glob('*.png'))

    patches_by_original = defaultdict(list)
    for patch_path in all_patch_files:
        match = re.match(r"(.+)_patch_x(\d+)_y(\d+)", patch_path.stem)
        if match:
            original_name, x, y = match.groups()
            patches_by_original[original_name].append((patch_path, int(x), int(y)))
    
    if not patches_by_original: 
        print(f"  [ERROR] Reconstruction failed: No patches found in '{test_image_dir}'")
        print(f"  - 或是 patch 檔名不符合 '..._patch_xNUM_yNUM.png' 格式。")
        return {}

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    overlay_vis_dir = results_path / "reconstruction_overlays"; overlay_vis_dir.mkdir(exist_ok=True)
    label_vis_dir = results_path / "reconstruction_visuals"; label_vis_dir.mkdir(exist_ok=True)

    original_patch_size = vis_params.get('original_patch_size', imgsz)
    if original_patch_size != imgsz:
        print(f"  - [Info] Reconstruction scaling enabled: Model input {imgsz}px -> Original patch {original_patch_size}px")
    else:
        print(f"  - [Info] Reconstruction scaling disabled (model_imgsz == original_patch_size == {imgsz}px)")

    for original_name, patches in tqdm(patches_by_original.items(), desc="Reconstructing and Visualizing"):
        original_img_path = next(original_images_dir.glob(f"{original_name}.*"), None)
        gt_mask_path = original_gt_dir / f"{original_name}.png"
        
        if not original_img_path or not original_img_path.exists():
            print(f"  [Warning] Skipping '{original_name}': Original image not found at '{original_images_dir}'")
            continue
        if not gt_mask_path.exists():
            print(f"  [Warning] Skipping '{original_name}': Original label mask not found at '{gt_mask_path}'")
            continue

        original_image = cv2.imread(str(original_img_path))
        gt_mask_cv = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        
        if original_image is None:
            print(f"  [Warning] Skipping '{original_name}': Failed to read original image.")
            continue
        if gt_mask_cv is None: 
            print(f"  [Warning] Skipping '{original_name}': Failed to read original GT mask.")
            continue

        gt_mask_binary = (gt_mask_cv > 0).astype(np.uint8)
        h, w, _ = original_image.shape
        
        all_mask_inputs = [] 
            
        for patch_path, x, y in patches:
            predict_params = {
                'source': str(patch_path),
                'imgsz': imgsz,
                'conf': vis_params['min_conf'],
                'boxes': False # [FIX v1.4] 強制關閉 bounding box 預測
            }
            results = model_adapter.predict(**predict_params)
            
            if not results or not results[0]:
                continue

            if architecture == 'yolo':
                 if results[0].masks is None: continue
                 mask_data_logits = results[0].masks.data
                 
                 # [FIX v1.4] YOLO 也可能因為 boxes=False 而沒有 conf
                 # 我們需要遍歷 mask (如果存在)
                 if mask_data_logits.shape[0] == 0: continue # 真的沒找到
                 
                 for i in range(mask_data_logits.shape[0]):
                    mask_logits_at_imgsz = mask_data_logits[i]
                    
                    if original_patch_size != imgsz:
                        mask_resized_to_patch = nn.functional.interpolate(
                            mask_logits_at_imgsz.unsqueeze(0).unsqueeze(0), 
                            size=(original_patch_size, original_patch_size), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                    else:
                        mask_resized_to_patch = mask_logits_at_imgsz
                    
                    # 嘗試獲取分數，如果沒有 (因為 boxes=False)，就用 0.95
                    score = results[0].boxes.conf[i].cpu().item() if (results[0].boxes is not None and len(results[0].boxes.conf) > i) else 0.95
                    
                    all_mask_inputs.append(
                        (mask_resized_to_patch, 
                         score,
                         x, y) 
                    )

            else:
                # 非 YOLO: 獲取機率圖
                if hasattr(results[0], 'pred_mask_prob_np') and results[0].pred_mask_prob_np is not None:
                    mask_prob_at_imgsz = results[0].pred_mask_prob_np
                else:
                    mask_prob_at_imgsz = results[0].pred_mask_np.astype(np.float32)
                
                if original_patch_size != imgsz:
                    mask_resized_to_patch_np = cv2.resize(
                        mask_prob_at_imgsz, 
                        (original_patch_size, original_patch_size), 
                        interpolation=cv2.INTER_LINEAR 
                    )
                else:
                    mask_resized_to_patch_np = mask_prob_at_imgsz
                
                mask_tensor_for_stitch = torch.from_numpy(mask_resized_to_patch_np).to(model_adapter.device)
                
                all_mask_inputs.append(
                    (mask_tensor_for_stitch, 
                     0.95, 
                     x, y)
                )
        
        if not all_mask_inputs:
            reconstructed_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            reconstructed_mask = stitch_masks(all_mask_inputs, (w, h), architecture)
        
        # [FIX v1.4] 移除 NMS 和 Bounding Box 相關邏輯
        # final_boxes, final_scores, final_labels = [], [], []
        # if all_detections_on_image: ... (整段移除)

        total_tp += np.sum(np.logical_and(reconstructed_mask, gt_mask_binary))
        total_tn += np.sum(np.logical_and(np.logical_not(reconstructed_mask), np.logical_not(gt_mask_binary)))
        total_fp += np.sum(np.logical_and(reconstructed_mask, np.logical_not(gt_mask_binary)))
        total_fn += np.sum(np.logical_and(np.logical_not(reconstructed_mask), gt_mask_binary))

        per_image_iou = calculate_iou(reconstructed_mask, gt_mask_binary)
        label_vis_image = np.zeros((h, w, 3), dtype=np.uint8)
        label_vis_image[gt_mask_binary == 1] = [0, 255, 0] # GT = 綠色 (您程式碼中的原始設定)
        label_vis_image[reconstructed_mask == 1] = [0, 0, 255] # Pred = 紅色
        label_vis_image[np.logical_and(gt_mask_binary, reconstructed_mask) == 1] = [0, 255, 255] # TP = 青色
        cv2.imwrite(str(label_vis_dir / f"{original_name}_iou_{per_image_iou:.4f}.png"), label_vis_image)
        
        overlay = original_image.copy(); alpha = vis_params['alpha']
        
        # [FIX v1.4] 修正疊加顏色 (與您日誌中的顏色一致)
        # BGR 格式
        color_tp = (0, 255, 255)   # TP: 青色 (Yellow in BGR)
        color_fp = (0, 0, 255)   # FP: 紅色
        color_fn = (255, 0, 0)   # FN: 藍色
        
        temp_overlay = np.zeros_like(original_image)
        temp_overlay[np.logical_and(reconstructed_mask, gt_mask_binary)] = color_tp
        temp_overlay[np.logical_and(reconstructed_mask, np.logical_not(gt_mask_binary))] = color_fp
        temp_overlay[np.logical_and(np.logical_not(reconstructed_mask), gt_mask_binary)] = color_fn
        
        final_overlay_image = cv2.addWeighted(temp_overlay, alpha, original_image, 1 - alpha, 0)
        
        # [FIX v1.4] 移除 draw_final_boxes 和 draw_segmentation_contours
        # 只儲存一張沒有 box 的 overlay
        cv2.imwrite(str(overlay_vis_dir / f"{original_name}_overlay_iou_{per_image_iou:.4f}.png"), final_overlay_image)


    # --- 迴圈結束，計算並回傳整體指標 ---
    
    # 1. 計算前景 (油汙) IoU
    iou_denominator_oil = total_tp + total_fp + total_fn
    iou_oil = total_tp / iou_denominator_oil if iou_denominator_oil > 0 else 0
    
    # 2. 計算背景 (Background) IoU
    #    背景的 TP 其實就是原本的 TN
    #    背景的 FP 就是原本的 FN (預測為背景，但其實是油汙)
    #    背景的 FN 就是原本的 FP (預測為油汙，但其實是背景)
    iou_denominator_bg = total_tn + total_fn + total_fp
    iou_bg = total_tn / iou_denominator_bg if iou_denominator_bg > 0 else 0
    
    # 3. 計算真正的 mIoU (Mean IoU)
    mean_iou = (iou_oil + iou_bg) / 2
    
    # 其他指標保持不變
    acc_denominator = total_tp + total_tn + total_fp + total_fn
    final_accuracy = (total_tp + total_tn) / acc_denominator if acc_denominator > 0 else 0
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    f1_denominator = precision + recall
    final_f1_score = 2 * (precision * recall) / f1_denominator if f1_denominator > 0 else 0

    print(f"\n--- [Info] Reconstruction Evaluation Complete ---")
    print(f"    Overall Accuracy: {final_accuracy:.4f}")
    print(f"    Overall F1-Score: {final_f1_score:.4f}")
    print(f"    IoU (Oil):        {iou_oil:.4f}")      # 顯示前景 IoU
    print(f"    IoU (Background): {iou_bg:.4f}")       # 顯示背景 IoU
    print(f"    Mean IoU (mIoU):  {mean_iou:.4f}")     # 顯示平均 IoU
    
    # 回傳修正後的指標
    return {
        "reconstruction_accuracy": f"{final_accuracy:.4f}",
        "reconstruction_f1_score": f"{final_f1_score:.4f}",
        "reconstruction_iou_oil": f"{iou_oil:.4f}",      # 新增
        "reconstruction_iou_bg": f"{iou_bg:.4f}",        # 新增
        "reconstruction_mean_iou": f"{mean_iou:.4f}"     # 現在這是真正的 mIoU
    }


def run_reconstruction_pipeline(model_adapter, dataset_config, save_dir, gradcam_config=None, reconstruction_config=None, explicit_original_root=None):
    """
    [V6.0 純淨版] 
    - 移除所有標準 Overlay 生成 (因為 run_reconstruction_evaluation 已做)。
    - 僅生成 vis_reconstruction_gradcam。
    - 使用與 Evaluation 完全相同的 Regex 與路徑邏輯，確保圖片位置正確。
    """
    if not gradcam_config or not gradcam_config.get('enabled', False):
        return # 如果沒開 Grad-CAM，這個函式現在沒事可做，直接返回

    print("\n[Reconstruction] 開始執行 Grad-CAM 重建 (V6.0)...")
    save_dir = Path(save_dir)
    
    vis_grad_dir = save_dir / "vis_reconstruction_gradcam"
    vis_grad_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Info] Saving Grad-CAM outputs to: {vis_grad_dir}")

    # 初始化 Grad-CAM
    try:
        from .gradcam_utils import GradCAM, compute_input_saliency
        gradcam_obj = GradCAM(model_adapter.model)
        enable_saliency = gradcam_config.get('saliency_map', False)
    except ImportError: return

    # --- 1. 準備路徑 (與 Evaluation 對齊) ---
    dataset_path = Path(dataset_config['path'])
    test_img_dir = dataset_path / 'images' / 'test'
    if not test_img_dir.exists():
        test_img_dir = dataset_path / 'images' / 'val'
    if not test_img_dir.exists(): return

    # --- 2. 準備 Original Dataset 路徑 ---
    orig_img_root = None
    if explicit_original_root:
        orig_img_root = Path(explicit_original_root) / 'images' / 'test'
        if not orig_img_root.exists():
            # 嘗試其它結構
            orig_img_root = Path(explicit_original_root) / 'images'
            if not orig_img_root.exists():
                orig_img_root = Path(explicit_original_root)

    if not orig_img_root or not orig_img_root.exists():
        print("  [Error] Cannot find original images root for Grad-CAM. Skipping.")
        return

    # --- 3. 掃描 Patch (使用與 Evaluation 完全相同的邏輯) ---
    all_patch_files = sorted(list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png')))
    patches_by_scene = defaultdict(list)
    
    for p_file in all_patch_files:
        # [關鍵] 使用與 Evaluation 一致的 Regex
        match = re.match(r"(.+)_patch_x(\d+)_y(\d+)", p_file.stem)
        if match:
            scene_id, x, y = match.groups()
            patches_by_scene[scene_id].append({'file': p_file, 'x': int(x), 'y': int(y)})
    
    if not patches_by_scene:
        print("  [Warning] No patches matched regex for Grad-CAM.")
        return

    model_imgsz = 512
    if hasattr(model_adapter, 'config') and 'imgsz' in model_adapter.config:
        model_imgsz = model_adapter.config['imgsz']

    # [Fix] 獲取原始 Patch 尺寸 (若有設定)
    original_patch_size = None
    if reconstruction_config and 'original_patch_size' in reconstruction_config:
        original_patch_size = int(reconstruction_config['original_patch_size'])
        print(f"  [Info] Grad-CAM Reconstruction using original_patch_size: {original_patch_size}")

    # --- 4. 逐一場景重建 ---
    for scene_id, patch_list in tqdm(patches_by_scene.items(), desc="Reconstructing Grad-CAM"):
        
        # 4.1 尋找原始大圖
        orig_file = next(orig_img_root.glob(f"{scene_id}.*"), None)
        if not orig_file: continue

        temp_img = cv2.imread(str(orig_file))
        if temp_img is None: continue
        max_h, max_w = temp_img.shape[:2]

        # 4.2 初始化畫布 (只為了 Grad-CAM)
        full_heatmap = np.zeros((max_h, max_w), dtype=np.float32)
        full_count_map = np.zeros((max_h, max_w), dtype=np.float32)
        full_saliency = None
        if enable_saliency:
            # 預設 3 通道 RGB
            full_saliency = np.zeros((3, max_h, max_w), dtype=np.float32)

        # 4.3 處理 Patch
        for info in patch_list:
            p_file, x, y = info['file'], info['x'], info['y']
            
            patch_img_bgr = cv2.imread(str(p_file))
            if patch_img_bgr is None: continue
            
            # 決定目標尺寸
            if original_patch_size:
                target_h, target_w = original_patch_size, original_patch_size
            else:
                target_h, target_w = patch_img_bgr.shape[:2]

            # 邊界檢查 (使用目標尺寸)
            h_valid = min(target_h, max_h - y)
            w_valid = min(target_w, max_w - x)
            if h_valid <= 0 or w_valid <= 0: continue

            # 更新計數 (用於重疊平均)
            full_count_map[y:y+h_valid, x:x+w_valid] += 1.0

            # 準備輸入 Tensor
            img_rgb = cv2.cvtColor(patch_img_bgr, cv2.COLOR_BGR2RGB)
            val_transform = A.Compose([
                A.Resize(model_imgsz, model_imgsz),
                A.Normalize(mean=(0.417, 0.417, 0.417), std=(0.267, 0.267, 0.267)),
                ToTensorV2()
            ])
            input_tensor = val_transform(image=img_rgb)['image'].unsqueeze(0).to(model_adapter.device)
            
            # (A) Global Grad-CAM
            heatmap = gradcam_obj.generate_cam(input_tensor)
            
            # [Fix] 確保 heatmap 是 2D
            if heatmap.ndim > 2:
                heatmap = np.mean(heatmap, axis=0)
            
            # 放大回 Patch 目標尺寸
            if heatmap.shape[:2] != (target_h, target_w): 
                heatmap = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 裁切有效區域並疊加
            heatmap_valid = heatmap[0:h_valid, 0:w_valid]
            full_heatmap[y:y+h_valid, x:x+w_valid] += heatmap_valid
            
            # (B) RGB Saliency
            if enable_saliency and full_saliency is not None:
                s_maps = compute_input_saliency(model_adapter.model, input_tensor)
                for c in range(min(3, s_maps.shape[0])):
                    sm = s_maps[c]
                    if sm.shape[:2] != (target_h, target_w): 
                        sm = cv2.resize(sm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    
                    sm_valid = sm[0:h_valid, 0:w_valid]
                    full_saliency[c, y:y+h_valid, x:x+w_valid] += sm_valid

        # 4.4 正規化與存檔
        full_count_map[full_count_map == 0] = 1.0
        
        # 存 Global Grad-CAM
        full_heatmap /= full_count_map
        vis_h = (full_heatmap * 255).astype(np.uint8)
        vis_h = cv2.applyColorMap(vis_h, cv2.COLORMAP_JET)
        cv2.imwrite(str(vis_grad_dir / f"{scene_id}_GradCAM.jpg"), vis_h, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        
        # 存 RGB Saliency
        if enable_saliency and full_saliency is not None:
            for c in range(3):
                full_saliency[c] /= full_count_map
                s_vis = (full_saliency[c] * 255).astype(np.uint8)
                s_vis = cv2.applyColorMap(s_vis, cv2.COLORMAP_JET)
                ch_name = ['Red', 'Green', 'Blue'][c]
                cv2.imwrite(str(vis_grad_dir / f"{scene_id}_Saliency_CH{c}_{ch_name}.jpg"), s_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

    print(f"[Reconstruction Grad-CAM] 完成。")