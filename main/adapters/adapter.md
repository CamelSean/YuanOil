# Adapter 說明手冊 (Adapter Manual)

## 1. 核心概念：適配器模式 (Adapter Pattern)

本框架的核心設計是「適配器模式」。`main_runner.py` (總控制器) 並不知道如何「具體」訓練一個 UNet 或 SegFormer。

`main_runner.py` 只會對 `training_module.py` 下達一個統一的指令：「根據 `experiments.yaml` 中的 `architecture: 'unet'` 設定，給我一個模型物件，並呼叫它的 `train()` 方法。」

`training_module.py` 中的 `get_model_adapter` 工廠函式，會去 `adapters/` 目錄中尋找**被 `@register_model('unet')` 裝飾器 註冊**的 `UnetAdapter` 類別，並將其回傳。

`Adapter` (適配器) 的職責，就是將**不同模型**（YOLO, UNet, SegFormer...）的**不同實作細節**，全部**封裝 (Wrap)** 起來，對外提供一組**完全統一的標準介面**。

## 2. 標準介面 (Standard Interface)

所有 `adapters/` 目錄下的**非 YOLO**適配器類別，都必須實作以下標準函式：

* **`__init__(self, config)`**:
    * **職責：** 建構模型。
    * **動作：** 讀取 `config` 字典（來自 `experiments.yaml`），特別是 `architecture_cfg` 和 `base_model`。
    * **來源：** 它會決定是從 `segmentation_models_pytorch`、`Hugging Face transformers`、`TorchVision` 載入模型，或是像 `paper_unet_adapter.py` 一樣從零開始手動建構。

* **`train(self, data, results_path, **train_params)`**:
    * **職責：** 執行完整的模型訓練。
    * **動作：** 實作完整的 PyTorch 訓練迴圈，包含資料增強、`DataLoader`、`AMP`、`Early Stopping`、儲存權重，並在結束後繪製 Loss/IoU 圖表。
    * **回傳：** 一個包含 `best_model_path` 的字典。

* **`predict(self, source, imgsz, **kwargs)`**:
    * **職責：** 對單一張圖片執行預測。
    * **動作：** 載入圖片、套用 `Normalize` 和 `Resize`、執行 `model.eval()` 和 `model(image)`、將預測結果放大回原始尺寸。
    * **回傳：** 一個**模仿 YOLO 結果的自訂結果物件** (例如 `UnetPredictionResult`)，該物件必須包含 `.masks`, `.boxes`, `.plot()` 等屬性，以供 `evaluation_module.py` 統一呼叫。
    * **[更新 v1.3] 回傳值規範：** 必須回傳一個自訂結果物件 (例如 UnetPredictionResult)，該物件必須包含：pred_mask_binary_np: 二值化 (0/1) 的遮罩 (供 evaluation_module 使用)。pred_mask_prob_np: 原始機率圖 (0.0~1.0) (供 reconstruction_module 進行加權拼貼)。
    * **參數 boxes=False:** 適配器應支援此參數，以關閉 Bounding Box 的生成 (YOLO 專用)。

* **`val(self, data, split='test', **kwargs)`**:
    * **職責：** 在 `val` 或 `test` 資料集上計算標準指標 (通常是 Pixel IoU)。
    * **動作：** 遍歷 `DataLoader`，累計 IoU。
    * **回傳：** 一個**模仿 YOLO 驗證結果的 `MockMetrics` 物件**，使其具有 `.seg.map50` 屬性，以供 `main_runner.py` 統一記錄。

## 3. 特殊適配器：`yolo_adapter.py`

`'yolo'` 是一個例外，它**不**包含自訂訓練迴圈。它是一個**純粹的包裝器** (Wrapper)，其 `train`, `predict`, `val` 函式只是單純地將指令轉發給 `ultralytics` 函式庫的 `self.model.train()`、`self.model.predict()` 和 `self.model.val()`。

## 4. 進階功能

`Adapter` 的設計使其可以輕鬆實作高度複雜的進階功能：

* **多通道輸入 (`in_channels`)**:
    * `unet_adapter_aux.py` 和 `paper_unet_adapter.py` 支援此功能。
    * 它們會讀取 `architecture_cfg.in_channels` 參數（例如 1 或 2）。
    * `SegmentationDataset` 會相應地只讀取灰階或 R/G 雙通道影像。
    * `train` 函式會動態調整 `Normalize` 的均值和標準差維度。
    * `__init__` 會將 `in_channels` 參數傳遞給模型建構函式。

* **輔助損失 (`aux_loss_weight`)**:
    * `unet_adapter_aux.py` 和 `segformer_adapter_aux.py` 支援此功能。
    * 它們會在 `train` 迴圈中，額外計算一個來自模型中間層的輔助損失，並將其加到總損失中：`total_loss = loss_main + (aux_loss_weight * loss_aux)`。
* **資料擴增**:
    * 在 train 函式中，讀取 YAML 中的 elastic_transform, coarse_dropout, gauss_noise 參數。
    * 動態建構 Albumentations 的轉換流程。
* **RS3Mamba 支援 (rs3mamba_adapter.py)**:
    * 處理 Mamba 特有的環境依賴 (mamba_ssm)。
    * 支援載入 vmamba 預訓練權重。
    * 處理特殊的張量維度需求。