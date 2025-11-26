# ======================================================================================
# ===           paper_unet_adapter.py (根據論文從零開始建構的 U-Net)                ===
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
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp # 仍然借用其 loss function
import pandas as pd
import matplotlib.pyplot as plt

from ..training_module import register_model

# ===================================================================
# 1. 從零開始建構 U-Net 模型
# ===================================================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PaperUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(PaperUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # factor 用於當 bilinear=False (使用 ConvTranspose2d) 時調整通道數
        # 當 bilinear=True (使用 Upsample) 時，factor=2，但您的 Up 模組設計
        # 會在 DoubleConv 中處理通道 (in_channels // 2)，所以我們主要關注 out_channels
        factor = 2 if bilinear else 1

        # Encoder: 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        self.down6 = Down(512, 1024 // factor) 
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)  # in = (bottleneck 1024//f) + (skip 512) = 512 + 512 = 1024 (假設 f=2)
        self.up2 = Up(512, 256 // factor, bilinear)  # in = (up1 out 512//f) + (skip 256) = 256 + 256 = 512
        self.up3 = Up(256, 128 // factor, bilinear)  # in = (up2 out 256//f) + (skip 128) = 128 + 128 = 256
        self.up4 = Up(128, 64 // factor, bilinear)   # in = (up3 out 128//f) + (skip 64) = 64 + 64 = 128
        self.up5 = Up(64, 32, bilinear)              # in = (up4 out 64//f) + (skip 32) = 32 + 32 = 64
        self.up6 = Up(48, 16, bilinear)           # in = (up5 out 32) + (skip 16) = 32 + 16 = 48  
        
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # 編碼器
        x1 = self.inc(x)     # out: 16
        x2 = self.down1(x1)  # out: 32
        x3 = self.down2(x2)  # out: 64
        x4 = self.down3(x3)  # out: 128
        x5 = self.down4(x4)  # out: 256
        x6 = self.down5(x5)  # out: 512
        x7 = self.down6(x6)  # out: 1024//f (Bottleneck)
        
        # 解碼器
        x = self.up1(x7, x6) # skip x6 (512)
        x = self.up2(x, x5) # skip x5 (256)
        x = self.up3(x, x4) # skip x4 (128)
        x = self.up4(x, x3) # skip x3 (64)
        x = self.up5(x, x2) # skip x2 (32)
        x = self.up6(x, x1) # skip x1 (16)
        
        logits = self.outc(x)
        return logits
# ===================================================================
# 2. 資料集與結果物件 (與其他 Adapter 相同)
# ===================================================================
# 放入您想要修改的分割 adapter 檔案中 (例如 paper_unet_adapter.py)
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, in_channels=3): # <--- [新增] 加入 in_channels 參數
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        self.transforms = transforms
        self.in_channels = in_channels # <--- [新增] 保存 in_channels
        print(f"SegmentationDataset initialized with in_channels={self.in_channels}") # 除錯訊息

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        try:
            # 先讀取 BGR 圖片
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                 raise ValueError(f"cv2.imread returned None for {img_path}")

            # --- [核心修改] 根據 in_channels 選擇通道 ---
            if self.in_channels == 1:
                 # 如果需要單通道，轉為灰階
                 image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                 # Albumentations 對單通道灰階圖的處理可能需要確認 Normalize 參數
                 # print(f"Debug: Loaded grayscale image shape: {image.shape}")
            elif self.in_channels == 2:
                 # 如果需要雙通道，選取 R 和 G 通道
                 # cv2 讀取是 BGR, 所以索引 2 是 R, 索引 1 是 G
                 image = image_bgr[:, :, [2, 1]] # <--- 選取 R 和 G 通道 (順序 GR)
                 # print(f"Debug: Loaded RG channels image shape: {image.shape}")
            elif self.in_channels == 3:
                 # 如果需要三通道，轉為 RGB (Albumentations 常用)
                 image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                 # print(f"Debug: Loaded RGB image shape: {image.shape}")
            else:
                 # 對於其他通道數，可以選擇報錯或進行其他處理
                 print(f"Warning: Unsupported in_channels={self.in_channels} requested. Defaulting to RGB.")
                 image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # 預設回退到 RGB

            # 讀取 Mask (保持不變)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None: mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
            mask = (mask > 0).astype(np.float32); mask = np.expand_dims(mask, axis=-1)

            # 應用 Albumentations 轉換
            if self.transforms:
                # 確保 Normalize 的 mean 和 std 維度與 in_channels 匹配
                # Albumentations 的 Normalize 會自動處理單通道或三通道
                # 對於雙通道，您可能需要在建立 transforms 時提供正確的 mean/std
                transformed = self.transforms(image=image, mask=mask)
                image_tensor = transformed['image'] # 轉換後為 Tensor
                mask_tensor = transformed['mask']
            else:
                 # 如果沒有轉換，需要手動轉 Tensor 並調整維度
                 # 這部分通常不會發生，因為 Resize 和 ToTensorV2 是必需的
                 image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() # HWC -> CHW
                 mask_tensor = torch.from_numpy(mask.transpose((2, 0, 1))).float()   # HWC -> CHW

            # 檢查 Tensor 的通道數是否符合預期
            # if image_tensor.shape[0] != self.in_channels:
            #      print(f"Warning: Tensor channel mismatch after transform! Expected {self.in_channels}, Got {image_tensor.shape[0]} for {img_path}")


            # 回傳 Tensor (permute mask tensor to CHW)
            return image_tensor, mask_tensor.permute(2, 0, 1) # Mask 原始是 HWC，轉為 CHW (1HW)

        except Exception as e:
            print(f"Error loading image/mask {img_path}: {e}")
            # 返回佔位符
            placeholder_imgsz = 512 # 假設
            # 嘗試從 transforms 獲取尺寸 (如果有的話)
            if self.transforms:
                 for t in self.transforms.transforms:
                      if isinstance(t, A.Resize): placeholder_imgsz = t.height; break
            return torch.zeros((self.in_channels, placeholder_imgsz, placeholder_imgsz)), torch.zeros((1, placeholder_imgsz, placeholder_imgsz))

class PaperUnetPredictionResult:
    
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
# 3. 核心適配器類別 (僅修改通道處理部分)
# ===================================================================
@register_model('paper_unet')
class PaperUnetAdapter(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
        print("--- Initializing Paper U-Net Adapter (Built from scratch) ---")
        self.config = exp_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        architecture_cfg = self.config.get('architecture_cfg', {})
        base_model_path = self.config.get('base_model')
        num_classes = self.config.get('dataset', {}).get('nc', 1)

        # [修改] 讀取並儲存 in_channels 設定，預設為 3
        self.in_channels = architecture_cfg.get('in_channels', 3)
        print(f"Paper UNet Config: in_channels={self.in_channels}, num_classes={num_classes}")

        # 建立模型時傳入讀取到的 in_channels
        self.model = PaperUNet(n_channels=self.in_channels, n_classes=num_classes)

        if base_model_path and Path(base_model_path).exists():
            print(f"權重來源: 本地檔案 '{base_model_path}'")
            try:
                 # 載入權重，如果通道數不同，strict=False 會嘗試載入匹配的部分
                 state_dict = torch.load(base_model_path, map_location=self.device, weights_only=True)
                 # 檢查輸入層權重通道數是否匹配
                 if 'inc.double_conv.0.weight' in state_dict and state_dict['inc.double_conv.0.weight'].shape[1] != self.in_channels:
                      print(f"  Warning: Input channel mismatch detected (model requires {self.in_channels}, checkpoint has {state_dict['inc.double_conv.0.weight'].shape[1]}). Input layer weights might be re-initialized if strict=False.")
                 # 使用 strict=False 允許載入部分權重 (例如，如果只有輸入層不匹配)
                 self.model.load_state_dict(state_dict, strict=False)
                 print("權重已嘗試載入 (strict=False)。")
            except Exception as e:
                 print(f"  載入權重時發生錯誤: {e}. 將使用隨機初始化權重。")
        else:
            print(f"權重來源: 從零開始隨機初始化")

        self.model.to(self.device)
        self.names = {i: n for i, n in enumerate(self.config.get('dataset', {}).get('names', ['oil']))}
        print(f"模型已移至設備: {self.device}")

    # (將此函式新增到 PaperUnetAdapter class 內部)
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
        # 保持您原有的 train 函式邏輯，僅修改 Normalize 和 Dataset 初始化
        from torch.cuda.amp import GradScaler
        import pandas as pd
        import matplotlib.pyplot as plt

        print("--- [Info] Starting Paper U-Net Training (Full Features with Plotting) ---")
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
        lr = train_params.get('lr0', 1e-4)

        architecture_cfg = self.config.get('architecture_cfg', {})
        loss_choice = architecture_cfg.get('loss_function', 'dice').lower()

        if loss_choice == 'focal':
            loss_fn = smp.losses.FocalLoss(mode='binary')
            print("  - [Info] Using FocalLoss")
        else:
            loss_fn = smp.losses.DiceLoss(mode='binary')
            print("  - [Info] Using DiceLoss (default)")

        # 資料增強設定 (保持不變)
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
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- 使用對應維度的 mean/std
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            A.Resize(imgsz, imgsz),
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- 使用對應維度的 mean/std
            ToTensorV2()
        ])

        # --- [修改] 建立 Dataset 時傳入 in_channels ---
        try:
             # 假設 SegmentationDataset 的 __init__ 已更新為接受 in_channels
             train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, train_transforms, in_channels=self.in_channels)
             val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, val_transforms, in_channels=self.in_channels)
        except FileNotFoundError as e:
             print(f"[Error] Dataset path not found: {e}")
             return None # 訓練無法繼續
        except TypeError as e:
             # 捕獲如果 SegmentationDataset 的 __init__ 沒有更新導致的錯誤
             if 'in_channels' in str(e):
                  print(f"[Error] SegmentationDataset initialization failed, does not accept in_channels parameter: {e}")
             else:
                  print(f"[Error] Unknown TypeError during Dataset initialization: {e}")
             return None
        except Exception as e:
             print(f"[Error] Unknown error loading dataset: {e}")
             return None

        # 檢查 Dataset 是否為空 (保持不變)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             print(f"[Error] Training ({train_img_dir}) or Validation ({val_img_dir}) dataset is empty.")
             return None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        # 後續訓練設定 (保持不變)
        use_amp = train_params.get('amp', False)
        accumulation_steps = train_params.get('gradient_accumulation_steps', 1)
        print(f"  - [Info] Training Config: LR={lr}, AMP={'Enabled' if use_amp else 'Disabled'}, Grad Accumulation={accumulation_steps}")

        scaler = GradScaler(enabled=use_amp)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        best_iou, best_model_path = -1.0, results_path/'weights'/'best.pt'; best_model_path.parent.mkdir(exist_ok=True, parents=True)

        patience = train_params.get('patience', 0)
        epochs_no_improve = 0
        if patience > 0: print(f"  - [Info] Early Stopping Enabled with patience: {patience}")

        history = []

        # --- 訓練和驗證迴圈 (保持您原始檔案的邏輯) ---
        for epoch in range(epochs):
            self.model.train(); running_loss = 0.0
            total_cm_train = np.zeros((2, 2), dtype=np.int64)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            # --- [開始] 保持原始訓練迴圈內部邏輯 ---
            for i, (images, masks) in enumerate(pbar):
                # 檢查 Dataset 返回的佔位符 (簡單檢查維度)
                if images.shape[-2:] != (imgsz, imgsz) or masks.shape[-2:] != (imgsz, imgsz):
                     print(f"  - [Warning] Skipping batch with unexpected shape. Image: {images.shape}, Mask: {masks.shape}")
                     continue

                images, masks = images.to(self.device), masks.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp): # 使用更新的 autocast
                    logits = self.model(images)
                    loss = loss_fn(logits, masks)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

                running_loss += loss.item() * accumulation_steps
                # 計算 Train IoU (使用 no_grad 避免影響梯度)
                with torch.no_grad():
                    preds_train_flat = (torch.sigmoid(logits) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                    masks_flat = masks.cpu().numpy().flatten().astype(np.uint8)
                    # 確保形狀匹配才計算混淆矩陣
                    if preds_train_flat.shape == masks_flat.shape:
                        cm_batch = confusion_matrix(masks_flat, preds_train_flat, labels=[0, 1])
                        # 確保 cm_batch 是 2x2
                        if cm_batch.shape == (2, 2):
                             total_cm_train += cm_batch

                processed_samples = i * batch_size + images.size(0)
                pbar.set_postfix(loss=running_loss / processed_samples if processed_samples > 0 else 0.0)
            # --- [結束] 保持原始訓練迴圈內部邏輯 ---

            avg_train_loss = running_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
            TN_t, FP_t, FN_t, TP_t = total_cm_train.ravel() if total_cm_train.sum() > 0 else (0, 0, 0, 0) # 处理 total_cm_train 可能为空的情况
            train_iou = TP_t / (TP_t + FP_t + FN_t + 1e-9)

            # --- 驗證迴圈 (保持您原始檔案的邏輯) ---
            self.model.eval(); total_cm_val = np.zeros((2, 2), dtype=np.int64)
            running_val_loss = 0.0
            val_samples_count = 0 # 記錄實際驗證的樣本數
            with torch.no_grad():
                # --- [開始] 保持原始驗證迴圈內部邏輯 ---
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    # 檢查無效數據
                    if images.shape[-2:] != (imgsz, imgsz) or masks.shape[-2:] != (imgsz, imgsz):
                        continue

                    images, masks = images.to(self.device), masks.to(self.device)
                    # 使用 AMP 進行驗證階段的前向傳播也可以
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                         logits = self.model(images)
                    running_val_loss += loss_fn(logits, masks).item() * images.size(0) # 乘以 batch size
                    val_samples_count += images.size(0)

                    preds_val_flat = (torch.sigmoid(logits) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                    masks_val_flat = masks.cpu().numpy().flatten().astype(np.uint8)
                    if preds_val_flat.shape == masks_val_flat.shape:
                         cm_batch_val = confusion_matrix(masks_val_flat, preds_val_flat, labels=[0, 1])
                         if cm_batch_val.shape == (2, 2):
                              total_cm_val += cm_batch_val
                # --- [結束] 保持原始驗證迴圈內部邏輯 ---

            avg_val_loss = running_val_loss / val_samples_count if val_samples_count > 0 else 0.0
            TN_v, FP_v, FN_v, TP_v = total_cm_val.ravel() if total_cm_val.sum() > 0 else (0,0,0,0)
            val_iou = TP_v / (TP_v + FP_v + FN_v + 1e-9)

            print(f"  - [Log] Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f}, Train IoU: {train_iou:.4f} | Val Loss: {avg_val_loss:.4f}, Val IoU: {val_iou:.4f}")
            history.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'train_iou': train_iou, 'val_iou': val_iou})

            # Early stopping 和模型儲存 (保持不變)
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

        # 繪圖和日誌 (保持不變)
        print("--- [Info] Training finished. Generating final result plots... ---")
        try:
            df = pd.DataFrame(history)
            if not df.empty: # 確保 history 不是空的
                 plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
                 plt.plot(df['epoch'], df['train_loss'], label='Train Loss'); plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
                 plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
                 plt.subplot(1, 2, 2)
                 plt.plot(df['epoch'], df['train_iou'], label='Train IoU'); plt.plot(df['epoch'], df['val_iou'], label='Validation IoU')
                 plt.title('IoU Curve'); plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True)
                 plt.tight_layout(); plt.savefig(results_path / 'training_curves.png'); plt.close()
                 df.to_csv(results_path / 'training_log.csv', index=False)
                 print(f"  - [Info] Final plots and logs saved to {results_path}")
            else:
                 print("  - [Warning] No history recorded, skipping plot generation.")
        except Exception as plot_err:
             print(f"  - [Error] Error generating final plots/log: {plot_err}")


        return {'best_model_path': str(best_model_path)}


    def predict(self, source, imgsz, **kwargs):
        """執行預測，確保使用正確的通道數和 Normalize"""
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
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- 使用對應的 mean/std
            ToTensorV2()
        ])

        try:
             # Albumentations 需要 HWC 格式輸入
             image_tensor = transforms(image=image_predict)['image'].unsqueeze(0).to(self.device)
        except Exception as e:
             print(f"[Error] Error during prediction transform for {source}: {e}")
             return [None] # 轉換失敗返回 None

        with torch.no_grad():
            self.model.eval()
            try:
                # 使用 AMP 進行預測
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    logits = self.model(image_tensor)
                # 放大回原始尺寸
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

            except Exception as e:
                print(f"[Error] Error during prediction forward pass for {source}: {e}")
                return [None] # 模型預測失敗

        # [FIX v1.3] 回傳兩個版本的遮罩
        return [PaperUnetPredictionResult(original_image, pred_mask_binary_np, pred_prob_np, **kwargs)]


    def val(self, data, split='test', **kwargs):
        """執行驗證，確保使用正確的通道數和 Normalize"""
        imgsz = kwargs.get('imgsz', 640)
        workers = kwargs.get('workers', 2) # 允許從 kwargs 傳遞 workers
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
        else: # 預設 RGB
            normalize_mean = (0.41733294, 0.41733294, 0.41733294)
            normalize_std = (0.26790292, 0.26790292, 0.26790292)

        transforms = A.Compose([
            A.Resize(imgsz, imgsz),
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- 使用對應的 mean/std
            ToTensorV2()
        ])

        # --- [修改] 建立 Dataset 時傳入 in_channels ---
        try:
             test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transforms, in_channels=self.in_channels)
        except FileNotFoundError as e:
             print(f"[Error] {split} dataset path not found: {e}")
             class MockMetrics: box = type('obj', (object,), {'mp':0.0, 'mr':0.0, 'map50': 0.0, 'map': 0.0})(); seg = box
             return MockMetrics()
        except Exception as e:
             print(f"[Error] Unknown error loading {split} dataset: {e}")
             class MockMetrics: box = type('obj', (object,), {'mp':0.0, 'mr':0.0, 'map50': 0.0, 'map': 0.0})(); seg = box
             return MockMetrics()


        if len(test_dataset) == 0:
             print(f"[Warning] {split} dataset is empty: {test_img_dir}")
             class MockMetrics: box = type('obj', (object,), {'mp':0.0, 'mr':0.0, 'map50': 0.0, 'map': 0.0})(); seg = box
             return MockMetrics()

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers) # 使用傳入的 workers

        total_iou = 0.0
        num_valid_samples = 0
        self.model.eval()
        with torch.no_grad():
            # --- [開始] 保持原始驗證迴圈內部邏輯 ---
            for images, masks in tqdm(test_loader, desc=f"Calculating Test Metrics ({split} split)"):
                # 檢查無效數據
                if images.shape[-2:] != (imgsz, imgsz) or masks.shape[-2:] != (imgsz, imgsz):
                     continue # 跳過

                images, masks = images.to(self.device), masks.to(self.device)
                # 使用 AMP 進行驗證
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                     logits = self.model(images)
                # 假設模型輸出尺寸與輸入/mask 尺寸相同
                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = torch.sum(preds * masks); union = torch.sum(preds) + torch.sum(masks) - intersection
                iou = ((intersection + 1e-6) / (union + 1e-6)).item()
                total_iou += iou
                num_valid_samples += 1
            # --- [結束] 保持原始驗證迴圈內部邏輯 ---

        avg_iou = total_iou / num_valid_samples if num_valid_samples > 0 else 0.0
        print(f"  - [Info] Validation ({split} split) calculated over {num_valid_samples} valid samples. Average IoU: {avg_iou:.4f}")

        # 返回模擬 YOLO 的指標物件 (保持不變)
        class MockMetrics:
            def __init__(self, iou):
                class Seg:
                    def __init__(self, iou): self.mp=0.0; self.mr=0.0; self.map50=iou; self.map=iou*0.9
                self.seg = Seg(iou); self.box = self.seg
        return MockMetrics(avg_iou)