import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

# 導入 segmentation-models-pytorch
import segmentation_models_pytorch as smp

# 從您的 training_module 導入註冊器
from ..training_module import register_model

# ===================================================================
# 1. & 2. 資料集與結果物件 (與其他 Adapter 相同)
# ===================================================================
class SegmentationDataset(Dataset):
    # [修改] 加入 in_channels 參數
    def __init__(self, image_dir, mask_dir, transforms=None, in_channels=3):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        self.transforms = transforms
        self.in_channels = in_channels # [修改] 保存 in_channels

    def __len__(self): return len(self.image_files)
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
             image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # smp 預設使用 RGB

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32); mask = np.expand_dims(mask, axis=-1)
        
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        return image, mask.permute(2, 0, 1)
    
class UnetPPPredictionResult:
    def __init__(self, original_image, pred_mask_binary, pred_mask_prob, **kwargs):
        self.original_image = original_image
        
        # [FIX v1.3] 儲存兩個版本的遮罩
        self.pred_mask_np = pred_mask_binary # (二值化, 0/1) 供 evaluation_module 和 plot 使用
        self.pred_mask_binary_np = pred_mask_binary # (明確命名)
        self.pred_mask_prob_np = pred_mask_prob     # (機率圖, 0.0~1.0) 供 reconstruction_module 使用
        
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
        if self.boxes:
            scale_x, scale_y = img.shape[1] / self.original_image.shape[1], img.shape[0] / self.original_image.shape[0]
            for box, conf in zip(self.boxes.xyxy, self.boxes.conf):
                x1, y1, x2, y2 = map(int, [box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2); cv2.putText(img, f"oil {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return img

# ===================================================================
# 3. 核心適配器類別
# ===================================================================
@register_model('unet++')
class UnetPPAdapter(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
        print("--- Initializing UNet++ Adapter (segmentation-models-pytorch) ---")
        self.config = exp_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        architecture_cfg = self.config.get('architecture_cfg', {})
        base_model_path = self.config.get('base_model')
        num_classes = self.config.get('dataset', {}).get('nc', 1)
        encoder_name = architecture_cfg.get('encoder_name', 'resnet34')
        encoder_weights = architecture_cfg.get('encoder_weights', 'imagenet')
        
        # [修改] 讀取 in_channels
        self.in_channels = architecture_cfg.get('in_channels', 3)
        print(f"  - Input channels set to: {self.in_channels}")
        
        # [修改] 如果 in_channels 不是 3，則不能使用 'imagenet' 預訓練權重
        if self.in_channels != 3 and encoder_weights == 'imagenet':
            print(f"  - [Warning] 'imagenet' weights require 3 channels, but {self.in_channels} are configured. Setting encoder_weights to None.")
            encoder_weights = None
            
        print(f"  - UNet++ Config: encoder_name='{encoder_name}', encoder_weights='{encoder_weights}'")

        # ⭐ 核心修改：將 smp.Unet 換成 smp.UnetPlusPlus
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.in_channels, # <--- [修改]
            classes=num_classes,
        )

        if base_model_path and Path(base_model_path).exists():
            print(f"  - Weight Source: Local file '{base_model_path}'")
            # [修改] 使用 strict=False 允許在通道數不匹配時載入
            try:
                self.model.load_state_dict(torch.load(base_model_path, map_location=self.device, weights_only=True), strict=False)
                print("  - Weights loaded successfully (strict=False).")
            except Exception as e:
                print(f"  - [Error] Failed to load weights: {e}")
        else:
            print(f"  - Weight Source: '{encoder_weights}' (from segmentation-models-pytorch)")

        self.model.to(self.device)
        self.names = {i: n for i, n in enumerate(self.config.get('dataset', {}).get('names', ['oil']))}
        print(f"  - Model moved to device: {self.device}")

    # (將此函式新增到 UnetPPAdapter class 內部)
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

    # (使用此函式替換舊的 train 函式)
    def train(self, data, results_path, **train_params):
        from torch.cuda.amp import GradScaler
        import pandas as pd
        import matplotlib.pyplot as plt

        print("--- [Info] Starting UNet++ Training (Full Features with Plotting) ---")
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])
        train_img_dir = base_path / data_config.get('train', 'images/train')
        train_mask_dir = base_path / 'labels' / Path(data_config.get('train', 'images/train')).name
        val_img_dir = base_path / data_config.get('val', 'images/val')
        val_mask_dir = base_path / 'labels' / Path(data_config.get('val', 'images/val')).name
        
        imgsz = train_params.get('imgsz', 512)
        batch_size = train_params.get('batch_size', 16)
        epochs = train_params.get('epochs', 100)
        workers = train_params.get('workers', 4)
        lr = train_params.get('lr0', 1e-4) # <--- [新增] 讀取 lr0 參數
        
        # [修改] 讀取包含 flip 在內的所有資料增強參數
        degrees, translate, scale, fliplr, flipud = [train_params.get(k, 0) for k in ['degrees', 'translate', 'scale', 'fliplr', 'flipud']]
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
            A.Resize(imgsz, imgsz), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            A.Resize(imgsz, imgsz), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])
        
        # --- [修改] 建立 Dataset 時傳入 in_channels ---
        train_loader = DataLoader(SegmentationDataset(train_img_dir, train_mask_dir, train_transforms, in_channels=self.in_channels), batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        val_loader = DataLoader(SegmentationDataset(val_img_dir, val_mask_dir, val_transforms, in_channels=self.in_channels), batch_size, num_workers=workers, pin_memory=True)

        use_amp = train_params.get('amp', False)
        accumulation_steps = train_params.get('gradient_accumulation_steps', 1)
        patience = train_params.get('patience', 0)
        print(f"  - [Info] Training Config: LR={lr}, AMP={'Enabled' if use_amp else 'Disabled'}, Grad Accumulation={accumulation_steps}, Patience={patience}")
        
        scaler = GradScaler(enabled=use_amp)
        loss_fn = smp.losses.DiceLoss(mode='binary')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr) # <--- [修改] 使用 lr 變數
        best_iou, best_model_path = -1.0, results_path/'weights'/'best.pt'; best_model_path.parent.mkdir(exist_ok=True, parents=True)
        epochs_no_improve = 0
        
        history = [] # <--- [新增] 初始化 history

        for epoch in range(epochs):
            self.model.train(); running_loss = 0.0
            total_cm_train = np.zeros((2, 2), dtype=np.int64)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for i, (images, masks) in enumerate(pbar):
                images, masks = images.to(self.device), masks.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    logits = self.model(images)
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
            
            # --- [新功能] 每 100 epoch 儲存一次圖表 ---
            if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
                self._save_epoch_plot(history, results_path, epoch + 1)
            # --- [新功能] 結束 ---

            if patience > 0 and epochs_no_improve >= patience:
                print(f"  - [Info] Early stopping triggered after {patience} epochs with no improvement.")
                break
        
        print("--- [Info] Training finished. Generating final result plots... ---")
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
        
        return {'best_model_path': str(best_model_path)}

    def predict(self, source, imgsz, **kwargs):
        original_image = cv2.imread(str(source))
        if original_image is None: return [None]
        h, w, _ = original_image.shape
        
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
            ToTensorV2()
        ])
        
        image_tensor = transforms(image=image_predict)['image'].unsqueeze(0).to(self.device) # <--- [修改]
        
        with torch.no_grad():
            self.model.eval()
            logits = self.model(image_tensor)
            upsampled_logits = nn.functional.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            
            # --- [FIX v1.3] ---
            # 1. 計算機率圖 (0.0 ~ 1.0)
            pred_prob_tensor = torch.sigmoid(upsampled_logits)
            
            # 2. 計算二值化圖 (0 或 1)
            pred_mask_binary_tensor = (pred_prob_tensor > 0.5)
            
            # 3. 轉換為 Numpy
            pred_prob_np = pred_prob_tensor.cpu().numpy().squeeze().astype(np.float32)
            pred_mask_binary_np = pred_mask_binary_tensor.cpu().numpy().squeeze().astype(np.uint8)
            # --- [FIX v1.3 END] ---
            
        # [FIX v1.3] 回傳兩個版本的遮罩
        return [UnetPPPredictionResult(original_image, pred_mask_binary_np, pred_prob_np, **kwargs)]
    
    def val(self, data, split='test', **kwargs):
        imgsz = kwargs.get('imgsz', 640)
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])
        test_img_dir = base_path / data_config.get(split, f'images/{split}')
        test_mask_dir = base_path / 'labels' / Path(data_config.get(split, f'images/{split}')).name
        
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
            A.Resize(imgsz, imgsz), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])
        
        # --- [修改] 建立 Dataset 時傳入 in_channels ---
        test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transforms, in_channels=self.in_channels)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        total_iou = 0.0; self.model.eval()
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Calculating Test Metrics"):
                images, masks = images.to(self.device), masks.to(self.device)
                logits = self.model(images)
                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = torch.sum(preds * masks); union = torch.sum(preds) + torch.sum(masks) - intersection
                total_iou += ((intersection + 1e-6) / (union + 1e-6)).item()
        avg_iou = total_iou / len(test_loader)
        class MockMetrics:
            def __init__(self, iou):
                class Seg:
                    def __init__(self, iou): self.mp=0.0; self.mr=0.0; self.map50=iou; self.map=iou*0.9
                self.seg = Seg(iou); self.box = self.seg
        return MockMetrics(avg_iou)