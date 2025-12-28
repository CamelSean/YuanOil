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
import segmentation_models_pytorch as smp
from ..training_module import register_model

# ===================================================================
# 輔助模組
# ===================================================================
class SimpleSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size, stride, padding)
    def forward(self, x): return self.conv(x)

# 自訂一個 Unet，使其 forward pass 回傳 encoder 的最後一個特徵
class UnetWithAux(smp.Unet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        """
        smp.Unet 的 encoder 會回傳一個 feature list, 
        list 中包含從 high-level 到 low-level 的特徵圖。
        我們需要將這個 list 作為單一參數傳遞給 decoder。
        """
        features = self.encoder(x)
        
        # ⭐ 核心修正：移除星號，將 features 作為單一列表參數傳遞
        decoder_output = self.decoder(features)
        
        masks = self.segmentation_head(decoder_output)
        
        # 回傳主輸出和 encoder 的最後一個 stage 的輸出 (bottleneck)
        return masks, features[-1]

# ===================================================================
# 1. & 2. 資料集與結果物件 (與標準 Unet 相同)
# ===================================================================
class SegmentationDataset(Dataset):
    # [修改] 加入 in_channels 參數
    def __init__(self, image_dir, mask_dir, transforms=None, in_channels=3):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        self.transforms = transforms
        self.in_channels = in_channels # [修改] 保存 in_channels
        # print(f"SegmentationDataset (unet_aux) initialized with in_channels={self.in_channels}") # 可選的除錯訊息

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        try:
            # 先讀取 BGR 圖片
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                 raise ValueError(f"cv2.imread returned None for {img_path}")

            # --- [修改] 根據 in_channels 選擇通道 ---
            if self.in_channels == 1:
                 image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            elif self.in_channels == 2:
                 image = image_bgr[:, :, [2, 1]] # 選取 R 和 G 通道 (順序 GR)
            elif self.in_channels == 3:
                 image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # 轉為 RGB
            else:
                 print(f"Warning: Unsupported in_channels={self.in_channels}. Defaulting to RGB.")
                 image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # 讀取 Mask (保持不變)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None: mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
            mask = (mask > 0).astype(np.float32); mask = np.expand_dims(mask, axis=-1)

            # 應用 Albumentations 轉換
            if self.transforms:
                transformed = self.transforms(image=image, mask=mask)
                image_tensor = transformed['image']
                mask_tensor = transformed['mask']
            else:
                 # 手動轉換 (通常不會走到這)
                 image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() if len(image.shape) == 3 else torch.from_numpy(image).unsqueeze(0).float()
                 mask_tensor = torch.from_numpy(mask.transpose((2, 0, 1))).float()

            return image_tensor, mask_tensor.permute(2, 0, 1) # Mask HWC -> CHW (1HW)

        except Exception as e:
            print(f"Error loading image/mask {img_path}: {e}")
            placeholder_imgsz = 512 # 預設或從 transform 獲取
            if self.transforms:
                 for t in self.transforms.transforms:
                      if isinstance(t, A.Resize): placeholder_imgsz = t.height; break
            return torch.zeros((self.in_channels, placeholder_imgsz, placeholder_imgsz)), torch.zeros((1, placeholder_imgsz, placeholder_imgsz))

class UnetPredictionResult:
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


@register_model('unet_aux')
class UnetAuxAdapter(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
        print("--- Initializing UNet Adapter with Auxiliary Loss ---")
        self.config = exp_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        architecture_cfg = self.config.get('architecture_cfg', {})
        base_model_path = self.config.get('base_model')
        num_classes = self.config.get('dataset', {}).get('nc', 1)
        encoder_name = architecture_cfg.get('encoder_name', 'resnet34')
        encoder_weights = architecture_cfg.get('encoder_weights', 'imagenet') # 預設使用預訓練

        # [修改] 讀取 in_channels，預設為 3
        self.in_channels = architecture_cfg.get('in_channels', 3)
        print(f"UNet_Aux Config: encoder_name='{encoder_name}', encoder_weights='{encoder_weights}', in_channels={self.in_channels}")

        # [修改] 建立模型時傳入 self.in_channels
        # 注意: smp 的 encoder_weights 是基於 ImageNet (3通道) 訓練的
        # 如果 in_channels 不是 3，encoder_weights 應設為 None 或自行處理權重載入
        current_encoder_weights = encoder_weights if self.in_channels == 3 else None
        if self.in_channels != 3 and encoder_weights == 'imagenet':
             print(f"  Warning: Requested in_channels={self.in_channels} but default encoder_weights='imagenet' requires 3 channels. Setting encoder_weights=None.")

        self.model = UnetWithAux(
            encoder_name=encoder_name,
            encoder_weights=current_encoder_weights, # 使用調整後的權重設定
            in_channels=self.in_channels, # <--- 傳入通道數
            classes=num_classes
        )

        # 輔助頭建立 (保持不變)
        aux_in_channels = self.model.encoder.out_channels[-1]
        self.aux_head = SimpleSegmentationHead(in_channels=aux_in_channels, num_classes=num_classes)
        print(f"輔助頭已建立: in_channels={aux_in_channels}, out_channels={num_classes}")

        # 載入權重 (保持不變)
        if base_model_path and Path(base_model_path).exists():
            print(f"權重來源: 本地檔案 '{base_model_path}' (僅載入主模型權重)")
            try:
                # 使用 strict=False 允許在通道數不同時載入部分權重
                state_dict = torch.load(base_model_path, map_location=self.device, weights_only=True)
                # 檢查 encoder 第一層權重通道數
                first_encoder_conv_key = None
                for key in state_dict.keys():
                    if key.startswith('encoder.') and 'conv' in key and 'weight' in key:
                         first_encoder_conv_key = key
                         break
                if first_encoder_conv_key and state_dict[first_encoder_conv_key].shape[1] != self.in_channels:
                     print(f"  Warning: Input channel mismatch (model={self.in_channels}, checkpoint={state_dict[first_encoder_conv_key].shape[1]}). Encoder input weights might not load correctly if strict=False.")

                self.model.load_state_dict(state_dict, strict=False)
                print("權重已嘗試載入 (strict=False)。")
            except Exception as e:
                 print(f"  載入權重時發生錯誤: {e}。模型權重可能未正確載入。")

        elif current_encoder_weights: # 如果使用了預設權重
            print(f"權重來源: '{current_encoder_weights}' (from smp, 僅主模型)")
        else: # 如果通道數不為3 或 未指定 base_model
             print(f"權重來源: 從零開始隨機初始化 (因 in_channels={self.in_channels} 或未提供 base_model)")


        self.model.to(self.device); self.aux_head.to(self.device)
        self.names = {i: n for i, n in enumerate(self.config.get('dataset', {}).get('names', ['oil']))}
        print(f"所有模型已移至設備: {self.device}")

    # (將此函式新增到 UnetAuxAdapter class 內部)
    def _save_epoch_plot(self, history, results_path, epoch, aux_loss_weight=0.4):
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
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss (Total)', linewidth=2)
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss (Main)')
            
            # 檢查是否有輔助損失
            if 'train_loss_main' in df.columns:
                plt.plot(df['epoch'], df['train_loss_main'], label='Train Loss (Main)', linestyle='--')
            if 'train_loss_aux' in df.columns:
                # 乘以權重以反映其對總 loss 的貢獻
                plt.plot(df['epoch'], df['train_loss_aux'] * aux_loss_weight, label=f'Train Loss (Aux * {aux_loss_weight:.2f})', linestyle=':')

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
        # 保持您上傳版本中的 train 函式邏輯，僅修改 Normalize 和 Dataset 初始化
        from torch.cuda.amp import GradScaler
        import pandas as pd
        import matplotlib.pyplot as plt

        print("--- [Info] Starting UNet Training (with Auxiliary Loss, Full Features) ---")
        aux_loss_weight = self.config.get('architecture_cfg', {}).get('aux_loss_weight', 0.4)
        print(f"  - [Info] Auxiliary Loss Weight: {aux_loss_weight}")
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

        # 資料增強 (保持您上傳的版本)
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
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- 使用對應的 mean/std
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            A.Resize(imgsz, imgsz),
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- 使用對應的 mean/std
            ToTensorV2()
        ])

        # --- [修改] 建立 Dataset 時傳入 in_channels ---
        try:
             train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, train_transforms, in_channels=self.in_channels)
             val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, val_transforms, in_channels=self.in_channels)
        except FileNotFoundError as e:
             print(f"[Error] Dataset path not found: {e}")
             return None
        except TypeError as e:
             if 'in_channels' in str(e):
                  print(f"[Error] SegmentationDataset initialization failed (likely not updated): {e}")
             else:
                  print(f"[Error] Unknown TypeError during Dataset initialization: {e}")
             return None
        except Exception as e:
             print(f"[Error] Unknown error loading dataset: {e}")
             return None

        if len(train_dataset) == 0 or len(val_dataset) == 0:
             print(f"[Error] Training ({train_img_dir}) or Validation ({val_img_dir}) dataset is empty.")
             return None


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        # 後續訓練設定 (保持您上傳的版本)
        use_amp = train_params.get('amp', False)
        accumulation_steps = train_params.get('gradient_accumulation_steps', 1)
        print(f"  - [Info] Training Config: LR={lr}, AMP={'Enabled' if use_amp else 'Disabled'}, Grad Accumulation={accumulation_steps}")

        scaler = GradScaler(enabled=use_amp)
        loss_fn = smp.losses.DiceLoss(mode='binary')
        # 優化器包含輔助頭參數 (保持不變)
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.aux_head.parameters()), lr=lr)
        best_iou, best_model_path = -1.0, results_path/'weights'/'best.pt'; best_model_path.parent.mkdir(exist_ok=True, parents=True)

        patience = train_params.get('patience', 0)
        epochs_no_improve = 0
        if patience > 0:
            print(f"  - [Info] Early Stopping Enabled with patience: {patience}")

        history = []

        # --- 訓練和驗證迴圈 (保持您上傳版本的邏輯，包括輔助損失計算) ---
        for epoch in range(epochs):
            self.model.train(); self.aux_head.train()
            running_loss, running_loss_main, running_loss_aux = 0.0, 0.0, 0.0
            total_cm_train = np.zeros((2, 2), dtype=np.int64)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            # --- [開始] 保持原始訓練迴圈內部邏輯 ---
            for i, (images, masks) in enumerate(pbar):
                if images.shape[-2:] != (imgsz, imgsz) or masks.shape[-2:] != (imgsz, imgsz): continue # 跳過無效數據

                images, masks = images.to(self.device), masks.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp): # 使用更新的 autocast
                    main_logits, aux_features = self.model(images)
                    loss_main = loss_fn(main_logits, masks)
                    aux_logits = self.aux_head(aux_features)
                    # 上採樣輔助 logits 到 mask 尺寸
                    upsampled_aux = nn.functional.interpolate(aux_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    loss_aux = loss_fn(upsampled_aux, masks)
                    total_loss = loss_main + (aux_loss_weight * loss_aux)
                    total_loss = total_loss / accumulation_steps

                scaler.scale(total_loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

                running_loss += total_loss.item() * accumulation_steps
                running_loss_main += loss_main.item() # 直接累加未加權的主 loss
                running_loss_aux += loss_aux.item() # 直接累加未加權的輔助 loss

                # 計算 Train IoU (基於主輸出)
                with torch.no_grad():
                     preds_train_flat = (torch.sigmoid(main_logits) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                     masks_flat = masks.cpu().numpy().flatten().astype(np.uint8)
                     if preds_train_flat.shape == masks_flat.shape:
                          cm_batch = confusion_matrix(masks_flat, preds_train_flat, labels=[0, 1])
                          if cm_batch.shape == (2, 2): total_cm_train += cm_batch


                processed_samples = i * batch_size + images.size(0)
                pbar.set_postfix(loss=running_loss / processed_samples if processed_samples > 0 else 0.0)
            # --- [結束] 保持原始訓練迴圈內部邏輯 ---

            # 計算平均 loss 和 train IoU
            num_train_batches = len(train_loader)
            avg_train_loss = running_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0 # 平均到每個樣本
            avg_train_loss_main = running_loss_main / num_train_batches if num_train_batches > 0 else 0.0 # 平均到每個 batch
            avg_train_loss_aux = running_loss_aux / num_train_batches if num_train_batches > 0 else 0.0 # 平均到每個 batch
            TN_t, FP_t, FN_t, TP_t = total_cm_train.ravel() if total_cm_train.sum() > 0 else (0,0,0,0)
            train_iou = TP_t / (TP_t + FP_t + FN_t + 1e-9)

            # --- 驗證迴圈 (保持您上傳版本的邏輯) ---
            self.model.eval(); self.aux_head.eval()
            total_cm_val = np.zeros((2, 2), dtype=np.int64)
            running_val_loss = 0.0
            val_samples_count = 0
            with torch.no_grad():
                # --- [開始] 保持原始驗證迴圈內部邏輯 ---
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    if images.shape[-2:] != (imgsz, imgsz) or masks.shape[-2:] != (imgsz, imgsz): continue

                    images, masks = images.to(self.device), masks.to(self.device)
                    # 驗證時通常只關心主輸出
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                         main_logits, _ = self.model(images)
                    running_val_loss += loss_fn(main_logits, masks).item() * images.size(0)
                    val_samples_count += images.size(0)

                    preds_val_flat = (torch.sigmoid(main_logits) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                    masks_val_flat = masks.cpu().numpy().flatten().astype(np.uint8)
                    if preds_val_flat.shape == masks_val_flat.shape:
                         cm_batch_val = confusion_matrix(masks_val_flat, preds_val_flat, labels=[0, 1])
                         if cm_batch_val.shape == (2, 2): total_cm_val += cm_batch_val
                # --- [結束] 保持原始驗證迴圈內部邏輯 ---

            avg_val_loss = running_val_loss / val_samples_count if val_samples_count > 0 else 0.0
            TN_v, FP_v, FN_v, TP_v = total_cm_val.ravel() if total_cm_val.sum() > 0 else (0,0,0,0)
            val_iou = TP_v / (TP_v + FP_v + FN_v + 1e-9)

            print(f"  - [Log] Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f}, Train IoU: {train_iou:.4f} | Val Loss: {avg_val_loss:.4f}, Val IoU: {val_iou:.4f}")
            # 記錄 history (保持不變)
            history.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'train_loss_main': avg_train_loss_main, 'train_loss_aux': avg_train_loss_aux, 'val_loss': avg_val_loss, 'train_iou': train_iou, 'val_iou': val_iou})

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
                self._save_epoch_plot(history, results_path, epoch + 1, aux_loss_weight=aux_loss_weight)
            # --- [新功能] 結束 ---

            if patience > 0 and epochs_no_improve >= patience:
                print(f"  - [Info] Early stopping triggered after {patience} epochs with no improvement.")
                break

        # 繪圖和日誌 (保持不變)
        print("--- [Info] Training finished. Generating final result plots... ---")
        try:
            df = pd.DataFrame(history)
            if not df.empty:
                 plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
                 plt.plot(df['epoch'], df['train_loss'], label='Train Loss (Total)', linewidth=2)
                 plt.plot(df['epoch'], df['train_loss_main'], label='Train Loss (Main)', linestyle='--')
                 # 乘以權重以反映其對總 loss 的貢獻
                 plt.plot(df['epoch'], df['train_loss_aux'] * aux_loss_weight, label=f'Train Loss (Aux * {aux_loss_weight:.2f})', linestyle=':')
                 plt.plot(df['epoch'], df['val_loss'], label='Validation Loss (Main)')
                 plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.ylim(bottom=0) # 限制 Y 軸最小值
                 plt.subplot(1, 2, 2)
                 plt.plot(df['epoch'], df['train_iou'], label='Train IoU'); plt.plot(df['epoch'], df['val_iou'], label='Validation IoU')
                 plt.title('IoU Curve'); plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True); plt.ylim(0, 1) # IoU 範圍 0-1
                 plt.tight_layout(); plt.savefig(results_path / 'training_curves.png'); plt.close()
                 df.to_csv(results_path / 'training_log.csv', index=False)
                 print(f"  - [Info] Final plots and logs saved to {results_path}")
            else:
                 print("  - [Warning] No history recorded, skipping plot generation.")
        except Exception as plot_err:
             print(f"  - [Error] Error generating final plots/log: {plot_err}")


        return {'best_model_path': str(best_model_path), 'best_val_score': best_iou}


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
            image_tensor = transforms(image=image_predict)['image'].unsqueeze(0).to(self.device)
        except Exception as e:
             print(f"[Error] Error during prediction transform for {source}: {e}")
             return [None]

        with torch.no_grad():
            self.model.eval() # 主模型設為 eval
            self.aux_head.eval() # 輔助頭也設為 eval (雖然預測不用)
            try:
                # 預測時通常只使用主模型的輸出
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                     logits, _ = self.model(image_tensor) # 接收兩個輸出，但只用第一個
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
        return [UnetPredictionResult(original_image, pred_mask_binary_np, pred_prob_np, **kwargs)]


    def val(self, data, split='test', **kwargs):
        """執行驗證，確保使用正確的通道數和 Normalize"""
        imgsz = kwargs.get('imgsz', 640)
        workers = kwargs.get('workers', 2)
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

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers)

        total_iou = 0.0
        num_valid_samples = 0
        self.model.eval(); self.aux_head.eval() # 兩個模型都設為 eval
        with torch.no_grad():
            # --- [開始] 保持原始驗證迴圈內部邏輯 ---
            for images, masks in tqdm(test_loader, desc=f"Calculating Test Metrics ({split} split)"):
                if images.shape[-2:] != (imgsz, imgsz) or masks.shape[-2:] != (imgsz, imgsz): continue

                images, masks = images.to(self.device), masks.to(self.device)
                # 驗證時只使用主輸出
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                     logits, _ = self.model(images)
                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = torch.sum(preds * masks); union = torch.sum(preds) + torch.sum(masks) - intersection
                iou = ((intersection + 1e-6) / (union + 1e-6)).item()
                total_iou += iou
                num_valid_samples += 1
            # --- [結束] 保持原始驗證迴圈內部邏輯 ---

        avg_iou = total_iou / num_valid_samples if num_valid_samples > 0 else 0.0
        print(f"  - [Info] Validation ({split} split) calculated over {num_valid_samples} valid samples. Average IoU: {avg_iou:.4f}")

        # 返回 MockMetrics (保持不變)
        class MockMetrics:
            def __init__(self, iou):
                class Seg:
                    def __init__(self, iou): self.mp=0.0; self.mr=0.0; self.map50=iou; self.map=iou*0.9
                self.seg = Seg(iou); self.box = self.seg
        return MockMetrics(avg_iou)