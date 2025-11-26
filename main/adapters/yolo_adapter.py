from ultralytics import YOLO
from ..training_module import register_model

@register_model('yolo')
class YoloAdapter:
    def __init__(self, config):
        """
        YOLO 適配器的建構函式。
        - config: 來自 experiments.yaml 的完整實驗設定字典。
        """
        print("--- Initializing YOLO Adapter ---")
        self.config = config
        # YOLO 的 base_model 可以是 .yaml (從頭訓練) 或 .pt (微調/預測)
        self.model = YOLO(config['base_model'])
        print(f"YOLO model '{config['base_model']}' loaded.")

        # [!!] 新增底下這 2 行來修正 'names' 屬性缺失錯誤
        dataset_cfg = self.config.get('dataset', {})
        self.names = {i: n for i, n in enumerate(dataset_cfg.get('names', ['oil']))}

    def train(self, data, results_path, **train_params):
        """
        執行 YOLO 模型的訓練。
        - data: 臨時生成的 data.yaml 檔案路徑。
        - results_path: 實驗結果的儲存路徑 (Path 物件)。
        - train_params: 來自 config['train'] 的訓練參數字典。
        """
        print("--- Starting YOLO Training ---")
        
        # YOLO 的 `train` 方法需要 `project` 和 `name` 參數來確定輸出目錄
        # 我們從 results_path 推導出來
        project_dir = results_path.parent
        exp_name = results_path.name

        # --- 參數名稱轉換 ---
        # 將通用的 'batch_size' 轉換為 YOLO 特定的 'batch'
        if 'batch_size' in train_params:
            train_params['batch'] = train_params.pop('batch_size')
        

        
        print(f"--- YOLO training parameters: data={data}, project={project_dir}, name={exp_name}, exist_ok=True, other_params={train_params} ---")

        self.model.train(
            data=data,
            project=str(project_dir),
            name=exp_name,
            exist_ok=True, # 允許覆寫已存在的實驗目錄
            **train_params  # 解包所有來自 YAML 的訓練參數
        )

        # YOLO 訓練完成後，最佳模型會儲存在 results_path/weights/best.pt
        best_model_path = results_path / 'weights' / 'best.pt'
        
        return {'best_model_path': str(best_model_path)}

    def predict(self, source, **predict_params):
        """
        執行 YOLO 模型的預測。
        - source: 圖片路徑。
        - predict_params: 來自 post_tests 的預測參數。
        """

        predict_params.pop('draw_bounding_boxes', None)

        return self.model.predict(source=source, **predict_params)

    def val(self, data, **val_params):
        """
        執行 YOLO 模型的驗證。
        - data: data.yaml 的路徑。
        - val_params: 驗證相關的參數。
        """
        # 確保 'split' 參數被正確傳遞
        if 'split' not in val_params:
            val_params['split'] = 'test'
            
        return self.model.val(data=data, **val_params)
