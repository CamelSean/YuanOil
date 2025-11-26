# 專案說明手冊 (Project Manual)
本文件旨在詳細說明此深度學習專案框架的結構、設定與使用方法，以便於快速理解、高效實驗與未來擴充。本手冊包含針對 SAR 影像的自動化困難樣本探勘 (HNM)、自適應裁切策略與 RS3Mamba 模型整合的最新說明。

---

## 1. 專案核心架構

本專案採用模組化設計，核心在於 `code/main` 資料夾，其主要元件如下：
-   **`main_runner.py`**: **專案總控制器**。它負責讀取 `experiments.yaml`，解析要執行的實驗，並根據設定依序啟動訓練、測試、評估等流程。
-   **`experiments.yaml`**: **實驗設定中心**。這是您定義所有實驗參數的地方，從模型架構、資料集路徑到訓練超參數，所有變動都應在此檔案中設定。
-   **`training_module.py`**: **模型訓練模組**。包含一個關鍵的工廠函式 `get_model_adapter()`，它會根據 `experiments.yaml` 中指定的 `architecture` 名稱，從 `adapters` 資料夾中抓取對應的模型適配器。
-   **`adapters/`**: **模型適配器目錄**。此目錄存放了所有不同模型架構的「包裝器」。每個適配器（如 `yolo_adapter.py`, `rs3mamba_adapter.py`, `segformer_adapter.py`）都遵循一個共同的介面，實作了 `train()` 和 `predict()` 等方法，使得 `main_runner.py` 可以用統一的方式來命令不同的模型進行訓練和預測。
-   **`reconstruction_module.py`**: **大圖重建模組**。負責將裁切後的預測圖塊（基於機率圖）拼回原始大圖尺寸，並計算包含 mIoU 在內的全圖指標。
-   **`evaluation_module.py`**: **評估模組**。計算像素級指標 (IoU, mIoU, F1) 並生成無框線的高透明度視覺化結果。

---

## 2. `experiments.yaml` 參數詳解
這是整個專案的指揮中心。以下是主要參數的說明。

### 2.1. 全域設定
-   `results_base_dir`: (字串) 所有實驗結果的根目錄。每個實驗會在此目錄下建立一個以時間戳和實驗名稱命名的子資料夾。
-   `excel_log_path`: (字串) 用於記錄每個實驗最終評估指標 (如 mAP, F1-score, mIoU) 的 Excel 檔案路徑。

### 2.2. 實驗 (`experiments`) 核心參數

-   `experiment_name`: (字串) 實驗的唯一名稱，將用於建立結果資料夾。
-   `mode`: (字串) `'train'` 或 `'test'`。
    -   `'train'`: 執行完整的訓練流程，並在結束後執行 `post_tests`。
    -   `'test'`: 只執行 `post_tests`，必須搭配 `base_model` 指定一個已經訓練好的模型權重路徑。
-   `architecture`: (字串) **[關鍵]** 指定要使用的模型架構。此名稱必須與 `training_module.py` 中 `MODEL_REGISTRY` 字典裡的鍵 (key) 完全對應。支援：`'yolo'`, `'unet'`, `'segformer'`, `'deeplabv3+'`, `'rs3mamba'` 等。
-   `base_model`: (字串) 預訓練模型的權重路徑 (`.pt` 檔案)。
    -   **對於 YOLO**:
        -   若要從頭開始訓練，需提供模型設定檔路徑 (例如 `yolov11-seg.yaml`)。**不可為空字串**。
        -   若提供 `.pt` 權重檔案路徑，則載入此權重進行微調 (fine-tuning) 或測試。
    -   **對於非 YOLO 模型**:
        -   若提供有效的 `.pt`/`.pth` 路徑，模型將直接載入。
        -   **優先級 1**: 如果提供一個有效的 `.pt` 或 `.pth` 檔案路徑，模型將直接從該檔案載入，忽略所有其他設定。
        -   **優先級 2**: 如果 `base_model` 為空 `''`，但 `architecture_cfg` 中定義了 `torchvision_weights` (例如 `'COCO_WITH_VOC_LABELS_V1'`)，則會從 TorchVision 下載對應的預訓練權重。
        -   **優先級 3**: 如果 `base_model` 為空 `''` 且沒有指定 `torchvision_weights`，模型將從零開始隨機初始化權重進行訓練。
-   `architecture_cfg`: (字典, 可選) 針對特定架構的額外設定。
    -   `in_channels`: (整數) 輸入通道數。例如 `1` (SAR 灰階) 或 `3` (RGB)。適配器會自動調整模型第一層。
    -   `loss_function`: `'focal'` 或 `'dice'` (僅部分 Adapter 如 `paper_unet`, `rs3mamba` 支援)。
    -   `pretrained_vmamba`: (字串) RS3Mamba 專用，指定 VMamba 預訓練權重路徑。
    -   `aux_loss_weight`: (浮點數) 輔助損失權重。(僅部分 Adapter 支援 如 `deeplabv3+_aux`)。
    -   `torchvision_weights: 'DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1'` 來指定從 TorchVision 下載的權重版本。

### 2.3. 資料集 (`dataset`) 參數

本節說明如何設定與準備資料集。

#### 2.3.1. 資料集路徑與結構

-   `path`: (字串) **訓練用資料集**的根路徑。此路徑下的圖片與標籤應為符合 `imgsz` 尺寸的預處理版本。
-   `nc`: (整數, 可選, 預設: 1) 資料集的類別數量。
-   `names`: (列表, 可選, 預設: `['oil']`) 資料集類別的名稱。

本框架預期的資料集結構如下，`path` 應指向 `dataset` 這個根目錄：
```
- dataset/
  - images/
    - train/
      - xxx.png
    - val/
      - yyy.png
    - test/
      - zzz.png
  - labels/
    - train/
      - xxx.png  # 用於語意分割的遮罩 (mask)
      - xxx.txt  # 用於 YOLO 的標籤 (label)
    - val/
    - test/
```
- **`images`**: 存放原始圖片。
- **`labels`**: 存放標籤檔案。

#### 2.3.2. 圖片尺寸與準備策略

建議採用 **「雙資料集策略」**：
1.  **訓練資料集** (`path`): 使用 `patch_method.py` 裁切並經過 `prepare_dataset.py` 縮放的小圖塊 (如 512x512)。
2.  **原始評估資料集** (`original_data_root`): 保留未裁切的原始大圖，用於 `post_tests` 中的重建評估。
3.  
**評估流程設計**: 模型預測會使用 `post_tests` 中 `dataset` 指定的圖片（通常是已縮放的測試圖）。然而，為了計算最精確的指標，後續處理流程如下：
    1.  系統會將模型輸出的低解析度預測遮罩 (尺寸為 `imgsz`) **放大回原始圖片的尺寸**。
    2.  使用這個放大後的預測結果，與「原始評估資料集」中的全尺寸標籤進行比較，來計算 IoU、F1-score 等指標。
    3.  所有需要基於原始尺寸的視覺化結果，如熱力圖 (`generate_heatmap`) 或最終分割圖，也會在這個階段生成。熱力圖尚未實作

#### 2.3.3. 遮罩圖 (Mask) 格式說明
對於語意分割任務，您在 `labels` 資料夾中提供的遮罩圖應遵循以下格式：
-   **格式**: 人類肉眼可直接辨識的 **RGB 圖檔** (例如黑白或彩色的 `.png` 檔案)。
-   **轉換**: **不需要**手動將其轉換為單通道的 8-bit 灰階圖。框架會在資料載入階段**自動處理**，將 RGB 遮罩圖轉換成模型訓練所需的單通道格式。

### 2.4. 訓練 (`train`) 參數

-   `epochs`: (整數) 訓練的總輪數。
-   `imgsz`: (整數) 訓練時輸入圖片的尺寸。對於標準模型，圖片和標籤都會被縮放到此尺寸。
-   `batch_size`: (整數) 批次大小。
-   `patience`: (整數) Early stopping 的耐心值。若驗證集損失在 `patience` 輪內沒有改善，則提前終止訓練。
-   `cls_weight`: (浮點數, for yolo ) 分類損失的權重。
-   `degrees`, `translate`, `scale`, `fliplr`, `flipud`: **幾何擴增** (必備，模擬視角變化)。
-   `elastic_transform`: **彈性失真** 模擬油汙受風浪影響的形狀扭曲。
-   `coarse_dropout`: **隨機遮擋** 強迫模型利用局部特徵。
-   `gauss_noise`: **高斯雜訊** ( 不太建議, 可選 )。增加對 SAR 斑點雜訊的抗性。
-   `workers`: (整數) 資料載入執行緒數。**注意**：注意：若遇到 CUDA error: invalid argument 或 Caught RuntimeError in pin memory thread，請設為 `0` (保證可行，但執行CPU速度下降)。

### 2.5. 後處理測試 (`post_tests`) 參數

這是一個列表，可以在訓練結束後執行多個不同的測試任務。

- - `test_name`: (字串) 該測試任務的名稱。
- - `dataset`: (字典) 指定此測試任務要使用的資料集。
- - `reconstruction`: (字典, 可選) **[通用功能]** 此為一項特殊工具，用於處理因原始圖片過大而需進行「裁切-訓練-重組」的流程。
    -   **工作流程**: 訓練時使用裁切後的小圖塊 (patch)，在 `post_tests` 階段，此功能會先對測試用的小圖塊進行預測，然後將這些預測結果**拼回**成一張與原始大圖相同尺寸的完整預測圖，最後再與原始大圖的標籤進行評估。
    -   `enabled`: (布林值) 是否啟用。
    -   `original_data_root`: (字串) **原始大圖資料集**的路徑。
    -   `original_patch_size`: (整數) **[新增][必要]** 原始圖塊 (Patch) 的尺寸。
        -   **說明**: 這是為了解決「縮放 (Resize) + 重建 (Reconstruct)」的問題。
        -   如果您的工作流包含 `patch_method.py`（例如，裁切為 1024x1024） 和 `prepare_dataset.py`（例如，縮放為 512x512），您必須在此提供**原始裁切的尺寸**（例如 `1024`）。
        -   `reconstruction_module` 會使用此參數，將模型輸出的 512x512 遮罩**放大回** 1024x1024，然後才拼接到正確的座標上。
    -   **[註] 補白 (Padding) 問題**: `patch_method.py` 在影像邊緣裁切時會使用補白。`reconstruction_module.py` 中的 `stitch_masks` 函式已內建**邊界檢查**，會自動裁切掉超出原始大圖範圍的補白區域，確保重建不會出錯。
- - `evaluation_on_original`: (字典, 可選) 
    -   **功能**: 控制是否在「原始高解析度」影像上進行指標計算 (Pixel Metrics) 和視覺化 (Categorized Predictions)。
    -   **程式碼行為**: `evaluation_module.py` 會檢查此設定。
    -   `enabled`: (布林值, 預設: `False`)
        -   若設為 `True`，系統會讀取 `original_data_root` 路徑下的高解析度圖片作為底圖，並將 `imgsz` 尺寸的預測遮罩放大回高解析度進行比較和儲存。
        -   若設為 `False` (預設)，系統會直接使用 `dataset.path` 中的低解析度圖塊（Patch）進行比較和儲存。
    -   `original_data_root`: (字串) 高解析度原始資料集的路徑。
    -   拼貼機制更新: 重建現在使用模型的原始機率圖 (0.0~1.0) 進行加權平均拼貼，而非二值化遮罩，大幅提升重疊區域的準確度。
- - `draw_bounding_boxes`: (布林值, 可選, 預設: `True`)
    -   **功能**: **[僅適用於非 YOLO 模型]** 控制純分割模型（如 UNet, DeepLab, SegFormer）是否在儲存 `TP`/`FP` 影像時，額外從預測遮罩中計算輪廓 (contour) 並繪製邊界框 (Bounding Box)。
    -   **程式碼行為**: `evaluation_module.py` 會將此參數傳遞給 `model_adapter.predict()`。各 `adapter` (如 `unet_adapter.py`) 會在 `PredictionResult` 物件中根據此開關決定是否產生 `self.boxes`。
    -   若設為 `False`，儲存的 `TP`/`FP` 影像將只包含遮罩疊加層，不包含邊界框。
- - `generate_heatmap`: (布林值, 可選) 是否生成熱力圖。此熱力圖是基於**原始圖片尺寸**的，會將預測結果放大後再進行繪製。**[尚未實作]**

---

## 3. 資料預處理工具 (Preprocessing)

### 3.1. 裁切工具 `preprocessing/patch_method.py` 

**目的**: 
-  將高解析度的原始大圖（例如 10000x10000）裁切成適合模型訓練的小圖塊（Patch），例如 1024x1024。
-  自動標籤生成: 自動將 PNG 遮罩轉換為 YOLO .txt 格式，並為負樣本生成空白的 .txt 檔案。

**主要功能**:
1.  **雙軌策略 (Hybrid Strategy)**：
    -   **Train/Val**: 使用 **自適應步長 (Adaptive Stride)**。
        -   油汙區域：使用 `OVERLAP_TRAIN_VAL` 進行密集採樣 (增加正樣本)。
        -   背景區域：強制 **無重疊 (Overlap=0)** (減少負樣本)。
    -   **Test**: 使用 **標準滑動窗口**。強制使用 `OVERLAP_TEST` (如 512) 進行均勻裁切，確保重建品質。
2.  **位移修正**：所有圖塊統一左上對齊，解決重建時的位移問題。
3.  **邊界補白**：對於無法完整覆蓋的邊緣區域，自動進行補白，確保所有圖塊尺寸一致。
4.  **正負樣本分離功能**: 可設定 SEPARATE_OUTPUT = True，腳本會自動檢查遮罩，將包含目標的圖塊 (Positive) 和純背景的圖塊 (Negative) 儲存到不同的資料夾（例如 train_pos 和 train_neg），這對於後續的困難樣本探勘至關重要。
5.  **多進程加速**：利用 `concurrent.futures` 平行處理，大幅縮短裁切時間。
6.  **檔名座標**：生成的檔名包含 `_x{座標}_y{座標}`，這是重建模組定位的依據。

### 3.2. `preprocessing/prepare_dataset.py`

用於將裁切後的圖塊 (如 1024) 縮放到模型輸入尺寸 (如 512)。

---
## 4. 進階工作流：自動化困難樣本探勘 (HNM)

位於 `preprocessing/run_automated_hnm_pipeline.py` 的全自動腳本，目標：在 SAR 影像中，「Lookalike (相似物)」(如海浪、低風速區) 常導致嚴重誤報。HNM 的目標是建立一個正負樣本數量平衡、且負樣本具有高度鑑別價值 (Hard Negative) 的高品質資料集。。

### 流程說明

#### 階段 1：訓練分類器 (Model v1)
-   **目標**：訓練一個輕量級二分類模型 (MobileNetV3)，讓它學會分辨「這張圖塊是不是油汙」。
-   **資料來源**:
    -   **類別 0 (Negative)**：來自 .../images/train_neg (純背景圖塊)。
    -   **類別 1 (Positive)**：來自 .../images/train_pos (含油汙圖塊) 中的所有圖片。
-   **篩選策略 (Balance Strategy)**：
    -   為了避免背景過多導致模型只會猜 0，使用參數 PHASE1_TRAIN_NEG_RATIO = 5。
    -   腳本會隨機從 train_neg 中抽出 N_pos * 5 張圖片參與訓練。
-   **Focal Loss**: 支援使用 Focal Loss 強化對少數類別的學習。
-   **比例限制**: 限制訓練時負樣本最多為正樣本的 N 倍 (如 1:5)，強迫模型敏感化。
-   **產出**：訓練好的分類器模型 (.pt) 與訓練曲線圖 (training_curves.png)。

#### 階段 2：探勘 (Mining)
-   **目標**：使用訓練好的分類器，對所有負樣本進行「體檢」，找出那些連分類器都會誤判的「困難樣本」。
-   **動作**：對 train_neg 和 val_neg 資料夾中的每一張圖片進行預測，得到其「是油汙的機率 (Prob)」。
-   **三元分割策略 (Ternary Split Strategy)**：根據預測機率，將負樣本分為三類：
    -   **Hard**: Prob $\ge$ 0.5 (這些圖片非常像油汙 (Lookalikes)，是我們最需要的負樣本。)。
    -   **Medium**: 0.1 $\le$ Prob < 0.5 (有點像，次要保留)。
    -   **Easy**: Prob < 0.1 (明顯背景，僅抽樣)。
-   **日誌 (Logging)**:腳本會輸出 hnm_2_mining_log.xlsx，詳細記錄每一張負樣本的預測機率和分類結果。
-   **產出**：將圖片分流複製到 hnm_2_mined/hard, hnm_2_mined/medium, hnm_2_mined/easy 資料夾中。

#### 階段 3：組裝黃金資料集 (Assembly)
-   **目標**：根據設定的「正負比例 (Ratio)」，動態組裝出多組最終的訓練資料集。
-   **參數控制**: TARGET_RATIOS = [1, 2, 3]：分別生成正負比 1:1, 1:2, 1:3 的三組資料集。
-   **功能**：支援 **多重比例 (Multi-Ratio)** 生成 (如 1:1, 1:2, 1:3)。
-   **邏輯**：
    -  **正樣本**：包含 100% 的 train_pos。
    -  **負樣本**：目標數量 = N_pos * Ratio。填充優先順序如下：
       -  優先 (Must)：放入 100% 的 Hard 樣本 (最有價值)。
       -  次要 (Should)：若還有空間，放入 Medium 樣本。
       -  補充 (Fill)：若還沒滿，從 Easy 樣本中隨機抽樣填滿 (防止模型忘記背景特徵)。
-   **資料完整性**：
    -   若 Hard 樣本本身就超過目標數量，則全數保留，比例會自然超過設定值 (这是预期的，但不太好)。
    -   Train 與 Val 採用「嚴格分組 (Strict Split)」，互不汙染。
-   **日誌 (Logging)**：輸出 hnm_3_gold_dataset_stats.xlsx，清楚列出每個 Ratio 資料集的正負樣本實際數量與分佈。
-   **產出**：
    -   生成多組不同比例的資料集，如HNM_Pipeline_v8_FinalStats/hnm_3_gold_dataset_ratio_1_1 等資料夾。
    -   這些路徑可直接填入 experiments.yaml 的 dataset.path 中使用。
    -   輸出 Excel 統計報表 (`_stats.xlsx`)。

---

## 5. 評估指標與視覺化更新

### 5.1. 新增指標
`evaluation_module` 與 `reconstruction_module` 現在計算完整的指標體系：
-   **IoU (Oil)**: 僅關注油汙類別的準確度 (最嚴格)。
-   **IoU (Background)**: 背景類別的準確度。
-   **mIoU**: `(IoU_Oil + IoU_Bg) / 2`。

### 5.2. 視覺化樣式
`categorized_predictions` 和 `reconstruction_overlays` 採用統一的新樣式：
-   **無框線**：移除所有 Bounding Box 和輪廓線。
-   **透明疊加**：使用高透明度顏色標示區域。
    -   **TP (真陽性)**: **青色 (Cyan)**
    -   **FP (偽陽性)**: **紅色 (Red)** - 誤報區域
    -   **FN (偽陰性)**: **藍色 (Blue)** - 漏抓區域

---

## 6. 新模型支援：RS3Mamba

本框架已整合 **RS3Mamba** (基於 VMamba 的遙測分割模型)。

### 使用需求
-   **環境**：必須使用 **CUDA 12.4** 與 **PyTorch 2.6.0** (為了解決 Mamba 依賴衝突)。
-   **依賴**：需安裝 `mamba-ssm`, `causal-conv1d`, `einops`, `mmcv`。

### 實驗設定範例
```yaml
- experiment_name: 'RS3Mamba_Train'
  mode: 'train'
  architecture: 'rs3mamba'
  architecture_cfg:
    in_channels: 1  # 支援單通道 SAR
    pretrained_vmamba: 'weights/vmamba_tiny_e292.pth'
    loss_function: 'focal' # 支援動態切換 Loss
  train:
    # ...
    workers: 0 # 避免 Pin memory 錯誤
```

---

## 7. 典型使用場景 (Scenarios)

### 場景 A: 從頭訓練一個 YOLO 模型
```yaml
- experiment_name: 'YOLO_from_scratch'
  mode: 'train'
  architecture: 'yolo'
  base_model: 'path/to/yolov11-seg.yaml' # 從頭訓練需提供 .yaml 設定檔
  imgsz: 512 # 這邊要用 目前有bug在測試階段沒讀取到train中的imgsz，會導致使用預設中的imgsz 640
  dataset:
    path: 'path/to/your/dataset'
  train:
    epochs: 100
    imgsz: 512
    batch_size: 32
```

### 場景 B: 微調一個 DeepLabV3+ 模型
- 此場景展示了兩個核心功能：`reconstruction` (處理裁切圖) 和「雙資料集評估策略」。
```yaml
- experiment_name: 'Finetune_DeepLab'
  mode: 'train'
  architecture: 'deeplabv3+'
  base_model: 'path/to/pretrained/best.pt' # 載入預訓練權重
  dataset:
    path: 'path/to/finetune/patches_512' # 1. 訓練/驗證使用 512x512 的裁切圖
  train:
    epochs: 50
    imgsz: 512
    batch_size: 16
```

### 場景 C: 訓練模型加上原始影像評估
```yaml
- experiment_name: 'DeepLab_and_Reconstruct'
  mode: 'train'
  architecture: 'deeplabv3+'
  base_model: 'path/to/pretrained/best.pt' 
  dataset:
    path: 'path/to/finetune/patches_512' # 1. 訓練/驗證使用 512x512 的裁切圖
  train:
    epochs: 50
    imgsz: 512
    batch_size: 16
  post_tests:
    - test_name: 'Eval_dataset'
      dataset:
        path: 'path/to/test/dataset1/patches_resize_512' # 2. 預測時使用 512x512 的測試裁切圖
      evaluation_on_original:
        enabled: True
        original_data_root: 'path/to/original/patches' # 3. 原始高裁切影像路徑 -> 會自己還原成resize前 該圖片大小並進行評估
    - test_name: 'Eval_dataset_version2'
      dataset:
        path: 'path/to/test/dataset2/patches_resize_512' 
      evaluation_on_original:
        enabled: True
        original_data_root: 'path/to/original/patches' 
```

### 場景 D: 訓練模型加上原始影像評估加上重建
```yaml
- experiment_name: 'DeepLab_and_Reconstruct'
  mode: 'train'
  architecture: 'deeplabv3+'
  base_model: 'path/to/pretrained/best.pt' 
  dataset:
    path: 'path/to/finetune/patches_512' # 1. 訓練/驗證使用 512x512 的裁切圖
  train:
    epochs: 50
    imgsz: 512
    batch_size: 16
  post_tests:
    - test_name: 'Eval_dataset'
      dataset:
        path: 'path/to/test/dataset1/patches_resize_512' # 2. 預測時使用 512x512 的測試裁切圖
      reconstruction:
        enabled: True
        original_data_root: 'path/to/original/patches' 
        original_patch_size: 1024 # 4. 先還原為3.原始高裁切圖塊尺寸 -> 需告訴還原大小 
      evaluation_on_original:
        enabled: True
        original_data_root: 'path/to/original/patches' # 3. 原始高解析度影像路徑 

```


---

