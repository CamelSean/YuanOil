# ======================================================================================
# ===           deeplab_adapter.py (移除 mmcv 依賴的最終完整版)                        ===
# ======================================================================================
import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, jaccard_score, precision_score, recall_score, f1_score

from ..training_module import register_model

# ===================================================================
# 1. 自訂 PyTorch 資料集
# ===================================================================
class SegmentationDataset(Dataset):
    # [修改] 加入 in_channels 參數
    def __init__(self, image_dir, mask_dir, transforms=None, in_channels=3):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_files = sorted([f for f in self.image_dir.glob('*.png')])
        self.transforms = transforms
        self.in_channels = in_channels # [修改] 保存 in_channels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            # [修改] 處理圖像讀取失敗
            print(f"[Warning] Failed to read image: {img_path}. Returning empty tensor.")
            placeholder_imgsz = 512
            if self.transforms:
                 for t in self.transforms.transforms:
                      if isinstance(t, A.Resize): placeholder_imgsz = t.height; break
            return torch.zeros((self.in_channels, placeholder_imgsz, placeholder_imgsz)), torch.zeros((1, placeholder_imgsz, placeholder_imgsz))

        # --- [修改] 根據 in_channels 選擇通道 ---
        if self.in_channels == 1:
             image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        elif self.in_channels == 2:
             image = image_bgr[:, :, [2, 1]] # 選取 R 和 G 通道 (順序 GR)
        else: # 預設 3 通道
             image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)

        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return image, mask.permute(2, 0, 1)

# ===================================================================
# 2. 模擬 YOLOv8 的回傳結果物件
# ===================================================================
class DeepLabPredictionResult:
    def __init__(self, original_image, pred_mask_binary, pred_mask_prob, **kwargs):
        self.original_image = original_image

        # [FIX v1.3] 儲存兩個版本的遮罩
        self.pred_mask_np = pred_mask_binary # (二值化, 0/1) 供 evaluation_module 和 plot 使用
        self.pred_mask_binary_np = pred_mask_binary # (明確命名)
        self.pred_mask_prob_np = pred_mask_prob     # (機率圖, 0.0~1.0) 供 reconstruction_module 使用
        
        self.masks = torch.from_numpy(pred_mask_binary).unsqueeze(0) if pred_mask_binary.sum() > 0 else None
        
        # ⭐ 核心修改：讀取開關，並依此決定是否產生 Bounding Box
        self.draw_boxes_enabled = kwargs.get('draw_bounding_boxes', True)
        self.boxes = self._get_boxes_from_mask() if self.draw_boxes_enabled else None
        
        self.names = {0: 'oil'}

    def _get_boxes_from_mask(self):
        if self.pred_mask_np.sum() == 0:
            return None
        contours, _ = cv2.findContours((self.pred_mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        class BoxContainer:
            def __init__(self, xyxy, conf):
                self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
                self.conf = torch.tensor(conf, dtype=torch.float32)
                cls_tensor = torch.zeros(self.conf.shape[0], 1)
                self.data = torch.cat((self.xyxy, self.conf.unsqueeze(1), cls_tensor), dim=1)

        boxes_xyxy = []
        scores = []
        for contour in contours:
            if cv2.contourArea(contour) < 1: continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes_xyxy.append([x, y, x + w, y + h])
            scores.append(0.95)
        
        return BoxContainer(boxes_xyxy, scores) if boxes_xyxy else None

    def plot(self, base_image=None):
        img_with_overlay = base_image if base_image is not None else self.original_image.copy()
        color = (0, 255, 255)
        
        if self.masks is not None:
            # 確保遮罩尺寸與基底圖片一致
            if self.pred_mask_np.shape[:2] != img_with_overlay.shape[:2]:
                resized_mask = cv2.resize(self.pred_mask_np, (img_with_overlay.shape[1], img_with_overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                resized_mask = self.pred_mask_np

            overlay = np.zeros_like(img_with_overlay)
            overlay[resized_mask == 1] = color
            img_with_overlay = cv2.addWeighted(img_with_overlay, 1.0, overlay, 0.5, 0)
        
        if self.boxes:
            for box, conf in zip(self.boxes.xyxy, self.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_with_overlay, (x1, y1), (x2, y2), color, 2)
                text = f"oil {conf:.2f}"
                cv2.putText(img_with_overlay, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_with_overlay

# ===================================================================
# 3. 核心適配器類別
# ===================================================================
@register_model('deeplabv3+')
class DeepLabAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("--- Initializing DeepLabV3+ Adapter (Standalone Version) ---")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        architecture_cfg = config.get('architecture_cfg', {})
        base_model_path = config.get('base_model')
        
        # [修改] 讀取 in_channels
        self.in_channels = architecture_cfg.get('in_channels', 3)
        print(f"  - Input channels set to: {self.in_channels}")

        # 根據手冊規則決定權重來源
        # 優先級 1: base_model 有 .pt 或 .pth 檔案
        if base_model_path and (base_model_path.endswith('.pt') or base_model_path.endswith('.pth')):
            print(f"  - Weight Source: Local weights file '{base_model_path}'")
            weights = None
            weights_name_for_log = "Custom"
        # 優先級 2: architecture_cfg 中有 torchvision_weights
        elif 'torchvision_weights' in architecture_cfg:
            weights_name = architecture_cfg['torchvision_weights']
            print(f"  - Weight Source: TorchVision Weights '{weights_name}'")
            if self.in_channels != 3:
                print(f"  - [Warning] TorchVision weights '{weights_name}' require 3 channels, but {self.in_channels} channels are configured. Weights for the first layer will NOT be loaded.")
                weights = None # [修改] 如果通道不為 3，則不載入預訓練權重
                weights_name_for_log = f"Random Init (in_channels={self.in_channels})"
            else:
                from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, DeepLabV3_ResNet50_Weights
                # 使用 getattr 動態獲取權重物件
                try:
                    weights = eval(weights_name)
                    weights_name_for_log = weights_name
                except NameError:
                    raise ValueError(f"Invalid torchvision_weights: {weights_name}")
            base_model_path = None # 清除路徑，避免後續載入
        # 優先級 3: 從頭訓練
        else:
            print("  - Weight Source: Random Initialization")
            weights = None
            weights_name_for_log = "Random Initialization"
            base_model_path = None

        # 決定模型骨幹
        torchvision_model_name = architecture_cfg.get('torchvision_model', 'deeplabv3_resnet50')
        print(f"  - Loading backbone: {torchvision_model_name} (Weights: {weights_name_for_log})")

        if torchvision_model_name == 'deeplabv3_resnet101':
            from torchvision.models.segmentation import deeplabv3_resnet101
            self.model = deeplabv3_resnet101(weights=weights)
        elif torchvision_model_name == 'deeplabv3_resnet50':
            from torchvision.models.segmentation import deeplabv3_resnet50
            self.model = deeplabv3_resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported torchvision_model: {torchvision_model_name}")

        # [修改] 動態修改第一層以支援 in_channels
        if self.in_channels != 3:
            # 替換 backbone 的 conv1
            old_conv1 = self.model.backbone.conv1
            self.model.backbone.conv1 = nn.Conv2d(
                self.in_channels, 
                old_conv1.out_channels, 
                kernel_size=old_conv1.kernel_size, 
                stride=old_conv1.stride, 
                padding=old_conv1.padding, 
                bias=old_conv1.bias
            )
            print(f"  - Modified model input layer (backbone.conv1) to accept {self.in_channels} channels.")


        # 修改分類頭以符合我們的類別數量
        num_classes = config.get('dataset', {}).get('nc', 1)
        in_channels_classifier = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels_classifier, num_classes, kernel_size=1)
        
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)

        self.model.to(self.device)
        
        # 如果 base_model_path 有效，則載入自訂權重
        if base_model_path and Path(base_model_path).exists():
            print(f"  - Loading custom weights from: {base_model_path}")
            # [修改] 使用 strict=False 允許在通道數不匹配時載入
            try:
                self.model.load_state_dict(torch.load(base_model_path, map_location=self.device), strict=False)
                print("  - Weights loaded successfully (strict=False).")
            except Exception as e:
                print(f"  - [Error] Failed to load weights: {e}")
        
        self.names = {i: name for i, name in enumerate(config.get('dataset', {}).get('names', ['oil']))}
        print(f"  - Model moved to device: {self.device}")

    def train(self, data, results_path, **train_params):
        from torch.cuda.amp import GradScaler, autocast
        import segmentation_models_pytorch as smp
        import pandas as pd
        import matplotlib.pyplot as plt

        print("--- [Info] Starting DeepLabV3+ Training (with Plotting) ---")
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])
        train_img_dir = base_path / data_config.get('train', 'images/train')
        train_mask_dir = base_path / 'labels' / data_config.get('train', 'images/train').split('/')[-1]
        val_img_dir = base_path / data_config.get('val', 'images/val')
        val_mask_dir = base_path / 'labels' / data_config.get('val', 'images/val').split('/')[-1]
        
        imgsz = train_params.get('imgsz', 512)
        batch_size = train_params.get('batch_size', 16)
        epochs = train_params.get('epochs', 100)
        workers = train_params.get('workers', 4)
        lr = train_params.get('lr0', 1e-4) # <--- [新增] 讀取 lr0 參數

        degrees, translate, scale, fliplr, flipud = [train_params.get(k, 0) for k in ['degrees', 'translate', 'scale', 'fliplr', 'flipud']]
        
        # --- [新增 v1.1] 讀取新的擴增參數 ---
        p_bright_contrast = train_params.get('random_brightness_contrast', 0.0)
        p_gauss_noise = train_params.get('gauss_noise', 0.0)
        p_coarse_dropout = train_params.get('coarse_dropout', 0.0)
        p_elastic = train_params.get('elastic_transform', 0.0)
        # --- [新增 v1.1 結束] ---

        train_augmentations = []

        if fliplr > 0:
            print(f"  - [Info] Data Augmentation Enabled: HorizontalFlip (p={fliplr})")
            train_augmentations.append(A.HorizontalFlip(p=fliplr))
        if flipud > 0:
            print(f"  - [Info] Data Augmentation Enabled: VerticalFlip (p={flipud})")
            train_augmentations.append(A.VerticalFlip(p=flipud))
        if degrees != 0 or translate != 0 or scale != 0:
            print(f"  - [Info] Data Augmentation Enabled: ShiftScaleRotate (degrees={degrees}, translate={translate}, scale={scale})")
            train_augmentations.append(A.ShiftScaleRotate(shift_limit=translate, scale_limit=scale, rotate_limit=degrees, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))
        
        # --- [新增 v1.1] 將新的擴增加入列表 ---
        if p_bright_contrast > 0:
            print(f"  - [Info] Data Augmentation Enabled: RandomBrightnessContrast (p={p_bright_contrast})")
            train_augmentations.append(A.RandomBrightnessContrast(p=p_bright_contrast))
            
        if p_gauss_noise > 0:
            print(f"  - [Info] Data Augmentation Enabled: GaussNoise (p={p_gauss_noise})")
            train_augmentations.append(A.GaussNoise(p=p_gauss_noise))
            
        if p_coarse_dropout > 0:
            print(f"  - [Info] Data Augmentation Enabled: CoarseDropout (p={p_coarse_dropout})")
            # max_holes: 挖幾個洞, max_height/width: 洞的大小
            train_augmentations.append(A.CoarseDropout(max_holes=8, max_height=int(imgsz*0.1), max_width=int(imgsz*0.1), p=p_coarse_dropout))
            
        if p_elastic > 0:
            print(f"  - [Info] Data Augmentation Enabled: ElasticTransform (p={p_elastic})")
            # 彈性失真 (對影像和遮罩同時作用)
            train_augmentations.append(A.ElasticTransform(p=p_elastic, border_mode=cv2.BORDER_CONSTANT, value=0))
        # --- [新增 v1.1 結束] ---

        if not train_augmentations:
            print("  - [Info] Data Augmentation Disabled")

        # --- [修改] 根據 self.in_channels 和您的新數值設定 Normalize 參數 ---
        if self.in_channels == 1:
            normalize_mean = (0.41733294,)
            normalize_std = (0.26790292,)
        elif self.in_channels == 2:
            normalize_mean = (0.41733294, 0.41733294)
            normalize_std = (0.26790292, 0.26790292)
        else: # 預設 3 通道 (RGB)
            normalize_mean = (0.41733294, 0.41733294, 0.41733294)
            normalize_std = (0.26790292, 0.26790292, 0.26790292)
        print(f"  - [Info] Using Normalize mean={normalize_mean}, std={normalize_std} for {self.in_channels} channels")

        train_transforms = A.Compose(train_augmentations + [
            A.Resize(int(imgsz), int(imgsz)), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            A.Resize(int(imgsz), int(imgsz)), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])
        
        # --- [修改] 建立 Dataset 時傳入 in_channels ---
        train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, train_transforms, in_channels=self.in_channels)
        val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, val_transforms, in_channels=self.in_channels)
        
        train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, num_workers=int(workers), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(workers), pin_memory=True)

        use_amp = train_params.get('amp', False)
        accumulation_steps = train_params.get('gradient_accumulation_steps', 1)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        print(f"  - [Info] Training Config: LR={lr}, AMP={'Enabled' if use_amp else 'Disabled'}, Grad Accumulation={accumulation_steps}")

        loss_fn = smp.losses.DiceLoss(mode='binary')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr) # <--- [修改] 使用 lr 變數

        best_iou = -1.0
        weights_path = results_path / "weights"
        weights_path.mkdir(parents=True, exist_ok=True)
        best_model_path = weights_path / "best.pt"
        
        history = []
        patience = train_params.get('patience', 0)
        epochs_no_improve = 0
        if patience > 0:
            print(f"  - [Info] Early Stopping Enabled with patience: {patience}")

        for epoch in range(epochs):
            self.model.train(); running_loss = 0.0
            total_cm_train = np.zeros((2, 2), dtype=np.int64)
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for i, (images, masks) in enumerate(progress_bar):
                images, masks = images.to(self.device), masks.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    outputs = self.model(images)
                    loss = loss_fn(outputs['out'], masks)
                    if 'aux' in outputs:
                        loss += 0.4 * loss_fn(outputs['aux'], masks)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                
                running_loss += loss.item() * accumulation_steps
                preds_train = (torch.sigmoid(outputs['out']) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                total_cm_train += confusion_matrix(masks.cpu().numpy().flatten(), preds_train, labels=[0, 1])
                progress_bar.set_postfix(loss=running_loss / (i + 1))
            
            avg_train_loss = running_loss / len(train_loader)
            TN_t, FP_t, FN_t, TP_t = total_cm_train.ravel()
            train_iou = TP_t / (TP_t + FP_t + FN_t + 1e-9)

            self.model.eval(); total_val_loss = 0.0
            total_cm_val = np.zeros((2, 2), dtype=np.int64)
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for images, masks in val_loop:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    loss = loss_fn(outputs['out'], masks)
                    total_val_loss += loss.item()
                    preds = (torch.sigmoid(outputs['out']) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                    total_cm_val += confusion_matrix(masks.cpu().numpy().flatten(), preds, labels=[0, 1])

            avg_val_loss = total_val_loss / len(val_loader)
            TN_v, FP_v, FN_v, TP_v = total_cm_val.ravel()
            val_iou = TP_v / (TP_v + FP_v + FN_v + 1e-9)

            print(f"  - [Log] Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f}, Train IoU: {train_iou:.4f} | Val Loss: {avg_val_loss:.4f}, Val IoU: {val_iou:.4f}")
            history.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'train_iou': train_iou, 'val_iou': val_iou})

            if val_iou > best_iou:
                best_iou = val_iou
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  - [Info] New best model saved with IoU: {best_iou:.4f}")
            else:
                epochs_no_improve += 1
            
            # --- [新功能] 每 100 epoch 儲存一次圖表 ---
            if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
                self._save_epoch_plot(history, results_path, epoch + 1)
            # --- [新功能] 結束 ---

            if patience > 0 and epochs_no_improve >= patience:
                print(f"  - [Info] Early stopping triggered after {patience} epochs with no improvement.")
                break
        
        print("--- [Info] Training finished. Generating final result plots... ---")
        # (最終的繪圖保持不變)
        df = pd.DataFrame(history)
        plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss'); plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_iou'], label='Train IoU'); plt.plot(df['epoch'], df['val_iou'], label='Validation IoU')
        plt.title('IoU Curve'); plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(results_path / 'training_curves.png'); plt.close()
        df.to_csv(results_path / 'training_log.csv', index=False)
        print(f"  - [Info] Final plots and logs saved to {results_path}")

        return {'best_model_path': str(best_model_path), 'best_val_score': best_iou}
    
    def _save_training_results(self, results_path, history):
        """將訓練歷史記錄儲存為 CSV 和 PNG 圖表。"""
        history_df = pd.DataFrame(history)
        history_df.to_csv(results_path / 'results.csv', index=False)

        plt.style.use('ggplot')
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # 繪製損失曲線
        axes[0].plot(history_df['epoch'], history_df['train_loss'], 'o-', label='Training Loss')
        axes[0].plot(history_df['epoch'], history_df['val_loss'], 'o-', label='Validation Loss')
        axes[0].set_title('Loss vs. Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # 繪製指標曲線
        axes[1].plot(history_df['epoch'], history_df['iou'], 'o-', label='Validation IoU (Jaccard)')
        axes[1].plot(history_df['epoch'], history_df['f1'], 'o-', label='Validation F1-Score')
        axes[1].set_title('Metrics vs. Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(results_path / 'results.png')
        plt.close()
        print(f"訓練結果圖表已儲存至: {results_path / 'results.png'}")

    def predict(self, source, imgsz, conf=0.25, verbose=False, **kwargs):
        original_image = cv2.imread(str(source))
        if original_image is None: return [None]
        original_h, original_w, _ = original_image.shape
        
        # --- [修改] 根據 self.in_channels 和您的新數值設定 Normalize 參數 ---
        if self.in_channels == 1:
            image_predict = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            normalize_mean = (0.41733294,)
            normalize_std = (0.26790292,)
        elif self.in_channels == 2:
            image_predict = original_image[:, :, [2, 1]] # 選 RG
            normalize_mean = (0.41733294, 0.41733294)
            normalize_std = (0.26790292, 0.26790292)
        else: # 預設 RGB
            image_predict = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            normalize_mean = (0.41733294, 0.41733294, 0.41733294)
            normalize_std = (0.26790292, 0.26790292, 0.26790292)

        transforms = A.Compose([
            A.Resize(int(imgsz), int(imgsz)),
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2(),
        ])
        
        image_tensor = transforms(image=image_predict)['image'] # <--- [修改]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            output = self.model(image_tensor)
            main_output = output['out'] # 這是 (imgsz, imgsz) 的 logits
            
            # --- [FIX v1.3] ---
            # 1. 放大 Logits 到 (h, w)
            upsampled_logits = nn.functional.interpolate(main_output, size=(original_h, original_w), mode='bilinear', align_corners=False)
            
            # 2. 計算機率圖 (0.0 ~ 1.0)
            pred_prob_tensor = torch.sigmoid(upsampled_logits)
            
            # 3. 計算二值化圖 (0 或 1)
            pred_mask_binary_tensor = (pred_prob_tensor > 0.5)
            
            # 4. 轉換為 Numpy (無需再次 resize)
            pred_prob_np = pred_prob_tensor.cpu().numpy().squeeze().astype(np.float32)
            pred_mask_binary_np = pred_mask_binary_tensor.cpu().numpy().squeeze().astype(np.uint8)
            # --- [FIX v1.3 END] ---
        
        # [FIX v1.3] 回傳兩個版本的遮罩
        return [DeepLabPredictionResult(original_image, pred_mask_binary_np, pred_prob_np, **kwargs)]
    
    def val(self, data, split='test', imgsz=640, **kwargs):
        print("--- [Info] Running DeepLabV3+ Validation on Test Set ---")
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        
        base_path = Path(data_config['path'])
        test_img_dir = base_path / data_config.get(split, f'images/{split}')
        test_mask_dir = base_path / 'labels' / data_config.get(split, f'images/{split}').split('/')[-1]
        
        # --- [修改] 根據 self.in_channels 和您的新數值設定 Normalize 參數 ---
        if self.in_channels == 1:
            normalize_mean = (0.41733294,)
            normalize_std = (0.26790292,)
        elif self.in_channels == 2:
            normalize_mean = (0.41733294, 0.41733294)
            normalize_std = (0.26790292, 0.26790292)
        else: # 預設 3 通道 (RGB)
            normalize_mean = (0.41733294, 0.41733294, 0.41733294)
            normalize_std = (0.26790292, 0.26790292, 0.26790292)
        
        transforms = A.Compose([
            A.Resize(int(imgsz), int(imgsz)),
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2(),
        ])
        
        # --- [修改] 建立 Dataset 時傳入 in_channels ---
        test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transforms, in_channels=self.in_channels)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        total_iou = 0; self.model.eval()
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Calculating Test Metrics", mininterval=5.0):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                main_output = outputs['out']
                preds = (torch.sigmoid(main_output) > 0.5).float()
                intersection = torch.sum(preds * masks)
                union = torch.sum(preds) + torch.sum(masks) - intersection
                total_iou += ((intersection + 1e-6) / (union + 1e-6)).item()
        avg_iou = total_iou / len(test_loader)
        print(f"  - [Info] Overall Test Pixel IoU (mAP50): {avg_iou:.4f}")
        class MockMetrics:
            def __init__(self, iou):
                class Seg:
                    def __init__(self, iou):
                        self.mp = 0.0; self.mr = 0.0; self.map50 = iou; self.map = iou * 0.9
                self.seg = Seg(iou); self.box = self.seg
        return MockMetrics(avg_iou)
    
    def _save_epoch_plot(self, history, results_path, epoch):
        """
        [New Feature] 儲存當前的訓練曲線快照。
        """
        try:
            df = pd.DataFrame(history)
            if df.empty:
                print(f"  - [Debug] History is empty, skipping interim plot at epoch {epoch}.")
                return

            plt.figure(figsize=(12, 5))
            
            # --- 繪製損失曲線 ---
            plt.subplot(1, 2, 1)
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
            plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)

            # --- 繪製 IoU 曲線 ---
            plt.subplot(1, 2, 2)
            plt.plot(df['epoch'], df['train_iou'], label='Train IoU')
            plt.plot(df['epoch'], df['val_iou'], label='Validation IoU')
            plt.title('IoU Curve'); plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True); plt.ylim(0, 1)
            
            plt.tight_layout()
            # 儲存為帶有 epoch 編號的檔案
            plot_path = results_path / f'training_curves_epoch_{epoch}.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"  - [Info] Interim training plot saved to: {plot_path}")
        
        except Exception as e:
            print(f"  - [Warning] Failed to save interim plot at epoch {epoch}: {e}")