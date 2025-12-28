# 檔案: main/adapters/rs3mamba_adapter.py

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
import segmentation_models_pytorch as smp 
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd

# 導入我們剛剛修改好的模型
from ..models.rs3mamba_core.RS3Mamba import RS3Mamba, load_pretrained_ckpt
from ..training_module import register_model

# ===================================================================
# 1. 資料集 (標準配置)
# ===================================================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, in_channels=3):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        self.transforms = transforms
        self.in_channels = in_channels 

    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            return torch.zeros((self.in_channels, 512, 512)), torch.zeros((1, 512, 512))
        
        # 1. 讀取與色彩空間轉換
        if self.in_channels == 1:
             image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # (H, W)
        elif self.in_channels == 2:
             image = image_bgr[:, :, [2, 1]] # (H, W, 2)
        else: 
             image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # (H, W, 3)

        # 2. 讀取 Mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32) # (H, W)
        
        # 3. Albumentations 增強
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 4. [FIX] 穩健的轉 Tensor 邏輯 (同時支援 Numpy 和 Tensor)
        
        # --- 處理影像 (Image) ---
        if isinstance(image, torch.Tensor):
            # 如果已經是 Tensor (代表 transforms 中有 ToTensorV2)
            # 形狀通常已經是 (C, H, W)，直接轉 float
            image_tensor = image.float()
        else:
            # 如果是 Numpy (代表 transforms 中沒有 ToTensorV2)
            if image.ndim == 2:
                # 灰階 (H, W) -> (1, H, W)
                image = image[np.newaxis, :, :] 
            else:
                # 多通道 (H, W, C) -> (C, H, W)
                image = image.transpose(2, 0, 1)
            image_tensor = torch.from_numpy(image).float()

        # --- 處理遮罩 (Mask) ---
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.float()
            # 如果 mask 是 Tensor 且是 (H, W)，補上 channel 維度
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            # Numpy 處理
            if mask.ndim == 2:
                mask = mask[np.newaxis, :, :]
            else:
                mask = mask.transpose(2, 0, 1)
            mask_tensor = torch.from_numpy(mask).float()

        return image_tensor, mask_tensor

# ===================================================================
# 2. 結果物件 (v1.3 機率圖支援)
# ===================================================================
class RS3MambaPredictionResult:
    def __init__(self, original_image, pred_mask_binary, pred_mask_prob, **kwargs):
        self.original_image = original_image
        self.pred_mask_np = pred_mask_binary        # 二值化遮罩 (供 evaluation)
        self.pred_mask_binary_np = pred_mask_binary # 二值化遮罩
        self.pred_mask_prob_np = pred_mask_prob     # 原始機率圖 (供 reconstruction)
        
        self.masks = torch.from_numpy(pred_mask_binary).unsqueeze(0) if pred_mask_binary.sum() > 0 else None
        self.draw_boxes_enabled = kwargs.get('draw_bounding_boxes', True)
        self.boxes = self._get_boxes_from_mask() if self.draw_boxes_enabled else None
        self.names = {0: 'oil'}

    def _get_boxes_from_mask(self):
        if self.pred_mask_np.sum() == 0: return None
        contours, _ = cv2.findContours((self.pred_mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        class BoxContainer:
            def __init__(self, xyxy, conf):
                self.xyxy = torch.tensor(xyxy, dtype=torch.float32); self.conf = torch.tensor(conf, dtype=torch.float32)
                self.data = torch.cat((self.xyxy, self.conf.unsqueeze(1), torch.zeros(self.conf.shape[0], 1)), dim=1)
        boxes_xyxy, scores = [], []
        for c in contours:
            if cv2.contourArea(c) < 1: continue
            x, y, w, h = cv2.boundingRect(c); boxes_xyxy.append([x, y, x + w, y + h]); scores.append(0.95)
        return BoxContainer(boxes_xyxy, scores) if boxes_xyxy else None

    def plot(self, base_image=None):
        img = base_image if base_image is not None else self.original_image.copy()
        if self.masks is not None:
            mask = cv2.resize(self.pred_mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros_like(img); overlay[mask == 1] = (0, 255, 255)
            img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)
        return img

# ===================================================================
# 3. Adapter 本體
# ===================================================================
@register_model('rs3mamba')
class RS3MambaAdapter(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
        print("--- [Info] Initializing RS3Mamba Adapter ---")
        self.config = exp_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        architecture_cfg = self.config.get('architecture_cfg', {})
        num_classes = self.config.get('dataset', {}).get('nc', 1)
        self.in_channels = architecture_cfg.get('in_channels', 3)
        print(f"  - Input channels set to: {self.in_channels}")
        
        base_model_path = self.config.get('base_model')

        # 1. 初始化模型 (傳入我們修改後支援 in_channels 的參數)
        self.model = RS3Mamba(
            in_channels=self.in_channels, 
            num_classes=num_classes,
            pretrained=True 
        )

        # 2. 載入 VMamba 預訓練權重 (若有指定)
        pretrained_vmamba_path = architecture_cfg.get('pretrained_vmamba', '')
        if pretrained_vmamba_path and Path(pretrained_vmamba_path).exists():
             print(f"  - Loading VMamba pretrained weights from: {pretrained_vmamba_path}")
             self.model = load_pretrained_ckpt(self.model, pretrained_vmamba_path)

        # 3. 載入 Fine-tuned 權重 (若有指定 base_model)
        if base_model_path and Path(base_model_path).exists():
            print(f"  - Loading checkpoint from: {base_model_path}")
            self.model.load_state_dict(torch.load(base_model_path, map_location=self.device), strict=False)

        self.model.to(self.device)
        self.names = {0: 'oil'}

    def _save_epoch_plot(self, history, results_path, epoch):
        try:
            df = pd.DataFrame(history)
            if df.empty: return
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
            plt.title('Loss Curve'); plt.legend(); plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.plot(df['epoch'], df['train_iou'], label='Train IoU')
            plt.plot(df['epoch'], df['val_iou'], label='Validation IoU')
            plt.title('IoU Curve'); plt.legend(); plt.grid(True)
            plt.tight_layout()
            plt.savefig(results_path / f'training_curves_epoch_{epoch}.png')
            plt.close()
        except Exception: pass

    def train(self, data, results_path, **train_params):
        from torch.cuda.amp import GradScaler
        import pandas as pd
        import matplotlib.pyplot as plt

        print("--- [Info] Starting RS3Mamba Training ---")
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])
        train_img_dir = base_path / data_config.get('train', 'images/train')
        train_mask_dir = base_path / 'labels' / Path(data_config.get('train', 'images/train')).name
        val_img_dir = base_path / data_config.get('val', 'images/val')
        val_mask_dir = base_path / 'labels' / Path(data_config.get('val', 'images/val')).name
        
        imgsz = train_params.get('imgsz', 512)
        batch_size = train_params.get('batch_size', 8)
        epochs = train_params.get('epochs', 100)
        workers = train_params.get('workers', 4)
        lr = train_params.get('lr0', 1e-4) 
        
        # 資料增強
        degrees, translate, scale, fliplr, flipud = [train_params.get(k, 0) for k in ['degrees', 'translate', 'scale', 'fliplr', 'flipud']]
        p_gauss_noise = train_params.get('gauss_noise', 0.0)
        p_coarse_dropout = train_params.get('coarse_dropout', 0.0)
        p_elastic = train_params.get('elastic_transform', 0.0)

        train_augmentations = []
        if fliplr > 0: train_augmentations.append(A.HorizontalFlip(p=fliplr))
        if flipud > 0: train_augmentations.append(A.VerticalFlip(p=flipud))
        if degrees != 0 or translate != 0 or scale != 0:
            train_augmentations.append(A.ShiftScaleRotate(shift_limit=translate, scale_limit=scale, rotate_limit=degrees, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))
        if p_gauss_noise > 0: train_augmentations.append(A.GaussNoise(p=p_gauss_noise))
        if p_coarse_dropout > 0: train_augmentations.append(A.CoarseDropout(max_holes=8, max_height=int(imgsz*0.1), max_width=int(imgsz*0.1), p=p_coarse_dropout))
        if p_elastic > 0: train_augmentations.append(A.ElasticTransform(p=p_elastic, border_mode=cv2.BORDER_CONSTANT, value=0))

        # Normalize 設定
        if self.in_channels == 1:
            normalize_mean, normalize_std = (0.41733294,), (0.26790292,)
        elif self.in_channels == 2:
            normalize_mean, normalize_std = (0.41733294, 0.41733294), (0.26790292, 0.26790292)
        else:
            normalize_mean, normalize_std = (0.41733294, 0.41733294, 0.41733294), (0.26790292, 0.26790292, 0.26790292)
            
        train_transforms = A.Compose(train_augmentations + [A.Resize(imgsz, imgsz), A.Normalize(mean=normalize_mean, std=normalize_std)])
        val_transforms = A.Compose([A.Resize(imgsz, imgsz), A.Normalize(mean=normalize_mean, std=normalize_std)])
        
        train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, train_transforms, in_channels=self.in_channels)
        val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, val_transforms, in_channels=self.in_channels)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        use_amp = train_params.get('amp', False)
        accumulation_steps = train_params.get('gradient_accumulation_steps', 1)
        scaler = GradScaler(enabled=use_amp)
        
        loss_choice = self.config.get('architecture_cfg', {}).get('loss_function', 'dice').lower()
        loss_fn = smp.losses.FocalLoss(mode='binary') if loss_choice == 'focal' else smp.losses.DiceLoss(mode='binary')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr) 
        best_iou, best_model_path = -1.0, results_path/'weights'/'best.pt'; best_model_path.parent.mkdir(exist_ok=True, parents=True)
        patience = train_params.get('patience', 0)
        epochs_no_improve = 0
        history = [] 

        for epoch in range(epochs):
            self.model.train(); running_loss = 0.0
            total_cm_train = np.zeros((2, 2), dtype=np.int64)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for i, (images, masks) in enumerate(pbar):
                images, masks = images.to(self.device), masks.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    logits = self.model(images)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    loss = loss_fn(logits, masks)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                
                running_loss += loss.item() * accumulation_steps
                preds_train = (torch.sigmoid(logits) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                total_cm_train += confusion_matrix(masks.cpu().numpy().flatten(), preds_train, labels=[0, 1])
                pbar.set_postfix(loss=running_loss / (i + 1))
            
            avg_train_loss = running_loss / len(train_loader)
            TN_t, FP_t, FN_t, TP_t = total_cm_train.ravel()
            train_iou = TP_t / (TP_t + FP_t + FN_t + 1e-9)

            self.model.eval(); total_cm_val = np.zeros((2, 2), dtype=np.int64)
            running_val_loss = 0.0
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    images, masks = images.to(self.device), masks.to(self.device)
                    logits = self.model(images)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    running_val_loss += loss_fn(logits, masks).item()
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                    total_cm_val += confusion_matrix(masks.cpu().numpy().flatten(), preds, labels=[0, 1])
            
            avg_val_loss = running_val_loss / len(val_loader)
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
            
            if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
                self._save_epoch_plot(history, results_path, epoch + 1)

            if patience > 0 and epochs_no_improve >= patience:
                print(f"  - [Info] Early stopping triggered after {patience} epochs.")
                break
        
        df = pd.DataFrame(history)
        if not df.empty:
            df.to_csv(results_path / 'training_log.csv', index=False)
            self._save_epoch_plot(history, results_path, epochs)

        return {'best_model_path': str(best_model_path), 'best_val_score': best_iou}

    def predict(self, source, imgsz, **kwargs):
        original_image = cv2.imread(str(source))
        if original_image is None: return [None]
        h, w, _ = original_image.shape
        
        if self.in_channels == 1:
            image_predict = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            normalize_mean, normalize_std = (0.41733294,), (0.26790292,)
        elif self.in_channels == 2:
            image_predict = original_image[:, :, [2, 1]]
            normalize_mean, normalize_std = (0.41733294, 0.41733294), (0.26790292, 0.26790292)
        else:
            image_predict = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            normalize_mean, normalize_std = (0.41733294, 0.41733294, 0.41733294), (0.26790292, 0.26790292, 0.26790292)

        transforms = A.Compose([A.Resize(int(imgsz), int(imgsz)), A.Normalize(mean=normalize_mean, std=normalize_std), ToTensorV2()])
        image_tensor = transforms(image=image_predict)['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            logits = self.model(image_tensor)
            upsampled_logits = nn.functional.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            
            # v1.3 實作：回傳機率圖 (0~1) 與 二值圖 (0/1)
            pred_prob_tensor = torch.sigmoid(upsampled_logits)
            pred_mask_binary_tensor = (pred_prob_tensor > 0.5)
            
            pred_prob_np = pred_prob_tensor.cpu().numpy().squeeze().astype(np.float32)
            pred_mask_binary_np = pred_mask_binary_tensor.cpu().numpy().squeeze().astype(np.uint8)
            
        return [RS3MambaPredictionResult(original_image, pred_mask_binary_np, pred_prob_np, **kwargs)]
    
    def val(self, data, split='test', **kwargs):
        imgsz = kwargs.get('imgsz', 640)
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])
        test_img_dir = base_path / data_config.get(split, f'images/{split}')
        test_mask_dir = base_path / 'labels' / Path(data_config.get(split, f'images/{split}')).name
        
        if self.in_channels == 1:
            normalize_mean, normalize_std = (0.41733294,), (0.26790292,)
        elif self.in_channels == 2:
            normalize_mean, normalize_std = (0.41733294, 0.41733294), (0.26790292, 0.26790292)
        else:
            normalize_mean, normalize_std = (0.41733294, 0.41733294, 0.41733294), (0.26790292, 0.26790292, 0.26790292)
            
        transforms = A.Compose([A.Resize(imgsz, imgsz), A.Normalize(mean=normalize_mean, std=normalize_std), ToTensorV2()])
        
        if not test_img_dir.exists():
             return MockMetrics(0.0)

        test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transforms, in_channels=self.in_channels)
        if len(test_dataset) == 0: return MockMetrics(0.0)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
        total_iou = 0.0; self.model.eval()
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Calculating Test Metrics"):
                images, masks = images.to(self.device), masks.to(self.device)
                logits = self.model(images)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = torch.sum(preds * masks); union = torch.sum(preds) + torch.sum(masks) - intersection
                total_iou += ((intersection + 1e-6) / (union + 1e-6)).item()
        avg_iou = total_iou / len(test_loader)
        return MockMetrics(avg_iou)

# 檔案: main/adapters/rs3mamba_adapter.py (最下方)

class MockMetrics:
    def __init__(self, iou):
        # 定義一個內部類別來模擬 YOLO 的指標物件結構
        class Seg:
            def __init__(self, iou):
                self.map50 = iou
                self.map = iou * 0.9
                self.mp = 0.0  # [FIX] 補上 Precision
                self.mr = 0.0  # [FIX] 補上 Recall
        
        self.seg = Seg(iou)
        self.box = self.seg # 讓 box 和 seg 指向同一個物件