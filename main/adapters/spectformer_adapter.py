import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp

# 導入註冊器
from ..training_module import register_model

# # 嘗試導入 SpectFormer 模型
# try:
#     from ..models.spectformer.spectformer import SpectFormer
# except ImportError:
#     print("[Error] Cannot import SpectFormer. Please ensure 'models/spectformer/spectformer.py' exists.")
#     pass
from ..models.spectformer.spectformer import SpectFormer

# ===================================================================
# 1. 模型定義 (SpectFormer + FPN Decoder) - 沿用您原本的邏輯
# ===================================================================
class SpectFormerSegBackbone(SpectFormer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward_features(self, x):
        B = x.shape[0]
        features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block: x = blk(x, H, W)
            norm = getattr(self, f"norm{i + 1}" if i != self.num_stages - 1 else f"norm{self.num_stages}")
            x_out = norm(x)
            x_out = x_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x_out)
        return features

class SimpleFPN_Decoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, num_classes=2):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        self.seg_head = nn.Conv2d(out_channels, num_classes, 1)
        
    def forward(self, inputs):
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] = laterals[i-1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear', align_corners=False)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        target_size = fpn_outs[0].shape[2:]
        fused = fpn_outs[0]
        for i in range(1, len(fpn_outs)):
            fused = fused + F.interpolate(fpn_outs[i], size=target_size, mode='bilinear', align_corners=False)
        return self.seg_head(fused)

class SpectFormerSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        arch_cfg = config.get('architecture_cfg', {})
        variant = arch_cfg.get('variant', 's')
        
        # 定義不同變體的參數
        specs = {
            's': {'embed_dims': [64, 128, 320, 448], 'depths': [3, 4, 6, 3]},
            'b': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 4, 12, 3]}
        }
        spec = specs.get(variant, specs['s'])
        
        self.backbone = SpectFormerSegBackbone(
            in_chans=3, # 強制 3 通道 (Pseudo-RGB)
            embed_dims=spec['embed_dims'],
            depths=spec['depths'],
            num_heads=[2, 4, 10, 14] if variant == 's' else [2, 4, 10, 16],
            num_classes=0
        )
        self.decoder = SimpleFPN_Decoder(in_channels_list=spec['embed_dims'], num_classes=2)

    def forward(self, x):
        input_shape = x.shape[2:]
        features = self.backbone.forward_features(x)
        logits = self.decoder(features)
        # 放大回原始尺寸
        return F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)

# ===================================================================
# 2. 資料集類別 (SpectFormerDataset) - 整合學長的邏輯
# ===================================================================
class SpectFormerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        self.transforms = transforms

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name

        # 讀取圖片，強制轉為 RGB 以配合 SpectFormer
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            # 錯誤處理：回傳全黑圖
            return torch.zeros((3, 512, 512)), torch.zeros((1, 512, 512)).long()
            
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32) 
        # Albumentations 需要 mask 有 channel 維度
        mask = np.expand_dims(mask, axis=-1)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # 回傳: image (3, H, W), mask (H, W) -> Long for CE Loss
        return image, mask.permute(2, 0, 1).squeeze(0).long()

# ===================================================================
# 3. 預測結果封裝 (SpectFormerPredictionResult) - 配合學長的 Evaluation
# ===================================================================
class SpectFormerPredictionResult:
    def __init__(self, original_image, pred_mask_binary, pred_mask_prob, **kwargs):
        self.original_image = original_image
        self.pred_mask_np = pred_mask_binary          # 給 Evaluation 用
        self.pred_mask_binary_np = pred_mask_binary   # 明確命名
        self.pred_mask_prob_np = pred_mask_prob       # 給 Reconstruction 用
        
        # 用於視覺化 (Boxes 預設關閉，為了與學長格式對齊)
        self.masks = None 
        self.boxes = None

# ===================================================================
# 4. Adapter 實作 (SpectFormerAdapter)
# ===================================================================
@register_model('spectformer')
class SpectFormerAdapter(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
        print("--- Initializing SpectFormer Adapter ---")
        self.config = exp_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.model = SpectFormerSegmentation(self.config)
        self.model.to(self.device)
        
        # 載入權重邏輯
        base_model_path = self.config.get('base_model')
        if base_model_path and Path(base_model_path).exists():
            print(f"  - Loading weights from: {base_model_path}")
            checkpoint = torch.load(base_model_path, map_location=self.device)
            # 處理權重匹配 (因為我們有加 Decoder)
            model_dict = self.model.state_dict()
            # 過濾掉形狀不符的 (例如 Decoder 層)
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            print(f"  - Loaded {len(pretrained_dict)}/{len(model_dict)} layers.")
        else:
            print("  - Initializing from scratch (or random).")

    def _save_epoch_plot(self, history, results_path, epoch):
        """儲存訓練曲線 (與學長功能對齊)"""
        try:
            df = pd.DataFrame(history)
            if df.empty: return
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1); plt.plot(df['epoch'], df['train_loss'], label='Train'); plt.plot(df['epoch'], df['val_loss'], label='Val'); plt.title('Loss'); plt.legend(); plt.grid(True)
            plt.subplot(1, 2, 2); plt.plot(df['epoch'], df['train_iou'], label='Train'); plt.plot(df['epoch'], df['val_iou'], label='Val'); plt.title('IoU'); plt.legend(); plt.grid(True)
            plt.savefig(results_path / f'training_curves_epoch_{epoch}.png'); plt.close()
        except Exception as e: print(f"Plotting error: {e}")

    def train(self, data, results_path, **train_params):
        print("--- [Info] Starting SpectFormer Training ---")
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])
        
        # 參數設定
        imgsz = train_params.get('imgsz', 512)
        batch_size = train_params.get('batch_size', 8)
        epochs = train_params.get('epochs', 100)
        lr = train_params.get('lr0', 1e-4)
        workers = train_params.get('workers', 4)

        # 資料增強 (對齊學長的設定)
        aug_list = []
        if train_params.get('fliplr', 0) > 0: aug_list.append(A.HorizontalFlip(p=train_params['fliplr']))
        if train_params.get('flipud', 0) > 0: aug_list.append(A.VerticalFlip(p=train_params['flipud']))
        
        # 正規化 (ImageNet Stats)
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
        
        train_transforms = A.Compose(aug_list + [A.Resize(imgsz, imgsz), A.Normalize(mean=norm_mean, std=norm_std), ToTensorV2()])
        val_transforms = A.Compose([A.Resize(imgsz, imgsz), A.Normalize(mean=norm_mean, std=norm_std), ToTensorV2()])

        # 建立 DataLoader (使用 SpectFormerDataset)
        train_ds = SpectFormerDataset(base_path / data_config['train'], base_path / 'labels' / Path(data_config['train']).name, train_transforms)
        val_ds = SpectFormerDataset(base_path / data_config['val'], base_path / 'labels' / Path(data_config['val']).name, val_transforms)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=workers, pin_memory=True)

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # 混合精度訓練
        use_amp = train_params.get('amp', False)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        accum_steps = train_params.get('gradient_accumulation_steps', 1)

        best_iou = 0.0
        history = []
        best_model_path = results_path / 'weights' / 'best.pt'
        best_model_path.parent.mkdir(exist_ok=True, parents=True)

        for epoch in range(epochs):
            self.model.train()
            run_loss = 0.0
            cm_train = np.zeros((2, 2), dtype=np.int64)
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for i, (imgs, masks) in enumerate(pbar):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(imgs) # (B, 2, H, W)
                    loss = criterion(logits, masks) / accum_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % accum_steps == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                
                run_loss += loss.item() * accum_steps
                
                # 計算 Train IoU
                preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
                gts = masks.cpu().numpy().flatten()
                cm_train += confusion_matrix(gts, preds, labels=[0, 1])
                pbar.set_postfix(loss=run_loss/(i+1))

            # Validation
            self.model.eval()
            val_loss = 0.0
            cm_val = np.zeros((2, 2), dtype=np.int64)
            with torch.no_grad():
                for imgs, masks in tqdm(val_loader, desc="[Val]"):
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    logits = self.model(imgs)
                    val_loss += criterion(logits, masks).item()
                    preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
                    gts = masks.cpu().numpy().flatten()
                    cm_val += confusion_matrix(gts, preds, labels=[0, 1])

            # 指標總結
            train_tn, train_fp, train_fn, train_tp = cm_train.ravel()
            val_tn, val_fp, val_fn, val_tp = cm_val.ravel()
            train_iou = train_tp / (train_tp + train_fp + train_fn + 1e-9)
            val_iou = val_tp / (val_tp + val_fp + val_fn + 1e-9)
            
            print(f"  - Epoch {epoch+1} | Train Loss: {run_loss/len(train_loader):.4f} IoU: {train_iou:.4f} | Val Loss: {val_loss/len(val_loader):.4f} IoU: {val_iou:.4f}")
            history.append({'epoch': epoch+1, 'train_loss': run_loss/len(train_loader), 'val_loss': val_loss/len(val_loader), 'train_iou': train_iou, 'val_iou': val_iou})

            # Save Best
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  - New Best Model Saved! IoU: {best_iou:.4f}")

            # Plotting
            if (epoch + 1) % 50 == 0: self._save_epoch_plot(history, results_path, epoch + 1)

        # Final Save
        df = pd.DataFrame(history)
        df.to_csv(results_path / 'training_log.csv', index=False)
        return {'best_model_path': str(best_model_path), 'best_val_score': best_iou}

    def predict(self, source, imgsz, **kwargs):
        # 讀取圖片
        img_bgr = cv2.imread(str(source))
        if img_bgr is None: return [None]
        
        # 轉換為 RGB (SpectFormer 需求)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
        transforms = A.Compose([A.Resize(int(imgsz), int(imgsz)), A.Normalize(mean=norm_mean, std=norm_std), ToTensorV2()])
        
        tensor_img = transforms(image=img_rgb)['image'].unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_img) # (1, 2, H, W)
            # 使用 Softmax 取得機率
            probs = torch.softmax(logits, dim=1)
            
            # Channel 1 是油汙的機率
            oil_prob_map = probs[0, 1, :, :].cpu().numpy().astype(np.float32)
            
            # 二值化
            oil_mask = (oil_prob_map > 0.5).astype(np.uint8)

        # 回傳學長格式的 Result 物件
        return [SpectFormerPredictionResult(img_bgr, oil_mask, oil_prob_map, **kwargs)]

    def val(self, data, split='test', **kwargs):
        # 為了滿足 evaluation_module 的需求，這裡實作一個簡易的 val
        # 實際上 evaluation_module 主要會呼叫 predict，這裡只是為了計算 mAP 等統計
        # 暫時回傳 Mock 物件即可，讓主程式能跑通
        class MockMetrics:
            def __init__(self):
                self.box = type('obj', (object,), {'map': 0.0, 'map50': 0.0, 'mp': 0.0, 'mr': 0.0})
                self.seg = type('obj', (object,), {'map': 0.0, 'map50': 0.0, 'mp': 0.0, 'mr': 0.0})
        return MockMetrics()