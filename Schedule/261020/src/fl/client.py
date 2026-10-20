# src/fl/client.py

import flwr as fl  
import torch  
from src.models.pytorch_model import DiabetesNet  
from src.train_utils import train, evaluate  
from src.seed import set_global_seed 

class DiabetesClient(fl.client.NumPyClient):  
    
    # ==========================================
    # 區塊 1：建構函式 (身份與裝備登記)
    # ==========================================
    # 【修改】接收 cid (客戶端 id) 以及 val_loader，捨棄 test_loader
    def __init__(self, cid: str, model: DiabetesNet, train_loader, val_loader):  
        self.cid = int(cid)  # 將 id 以整數保存，方便後續伺服器排序
        self.model = model   # 綁定本地端的神經網路模型
        self.train_loader = train_loader  
        self.val_loader = val_loader  # 嚴格遵守指南：調參期間只用驗證集 (val)，不用測試集 (test)

    # ==========================================
    # 區塊 2：權重翻譯官 (PyTorch Tensor <-> NumPy Array)
    # ==========================================
    def get_parameters(self, config):
        # 【上傳】抽出本地模型訓練後的權重 (state_dict)，並將 PyTorch 張量 (Tensor) 轉為 NumPy 格式
        # 因為 Flower 伺服器在網路上傳輸時，只看得懂 NumPy 陣列
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # 【下載】將伺服器傳來的 NumPy 權重陣列，跟本地神經網路的各個神經層配對
        params_dict = zip(self.model.state_dict().keys(), parameters)
        
        # 將配對好的 NumPy 陣列轉回 PyTorch 張量，組合成完整的模型權重字典
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        
        # 把這套新的「全域權重」硬生生覆寫進本地模型，作為這一輪訓練的起跑點
        self.model.load_state_dict(state_dict, strict=True)


    # ==========================================
    # 區塊 3：接收指令與在地訓練
    # ==========================================
    def fit(self, parameters, config):
        # 【防呆】：由執行種子、回合數與客戶端 id 計算出獨一無二的動態呼叫種子
        # 這能確保每一輪、每一個 Client 都有一個專屬且絕對固定的隨機亂數表
        server_round = config["server_round"]
        base_seed = config["seed"]
        call_seed = base_seed + 1000 * server_round + self.cid
        
        # 在 fit 內部重新設定全域種子，突破 Flower 底層每次呼叫會重新建立獨立進程的限制
        set_global_seed(call_seed)
        
        # 重設訓練 loader 的產生器種子，確保每輪抽籤洗牌 (shuffle) 的順序 100% 可重現
        self.train_loader.generator.manual_seed(call_seed)
        
        # 換上伺服器下發的最新全域權重
        self.set_parameters(parameters)
        
        # 【嚴格取值】：直接從 config 讀取伺服器傳來的參數，故意不使用 get() 給預設值。
        # 這樣如果伺服器忘記傳送 local_epochs 或 lr，程式會直接當掉報錯，而不會默默用預設值跑完。
        epochs = config["local_epochs"]
        lr = config["lr"]
        
        # 呼叫 train_utils.py 的 train 函式，利用本地病患資料進行微調
        self.model = train(self.model, self.train_loader, epochs=epochs, lr=lr)
        
        # 回傳：更新後的 NumPy 權重、本地訓練樣本數、包含 cid 的字典 (供伺服器排序用)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"cid": self.cid}


    # ==========================================
    # 區塊 4：模擬考與回報成績
    # ==========================================
    def evaluate(self, parameters, config):
        # 換上伺服器下發的最新全域權重
        self.set_parameters(parameters)
        
        # 呼叫 evaluate 函式，注意這裡傳入的是 val_loader (驗證集)
        loss, metrics = evaluate(self.model, self.val_loader)
        
        # 將指標轉為一般 Python 浮點數 (避免傳輸報錯)，並強制塞入客戶端 id
        metrics_float = {k: float(v) for k, v in metrics.items()}
        metrics_float["cid"] = self.cid
        
        # 回傳：平均 loss、驗證樣本數、浮點數 metrics 字典給 server.py
        return float(loss), len(self.val_loader.dataset), metrics_float