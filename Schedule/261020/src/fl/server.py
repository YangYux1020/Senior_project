# src/fl/server.py

import pandas as pd
import os 
import flwr as fl  
import numpy as np  
from flwr.common import Metrics  
from typing import List, Tuple  
from src.models.pytorch_model import DiabetesNet
from src.seed import set_global_seed

# ==========================================
# 區塊 1：加權平均評分系統
# ==========================================
# 目的：避免各分區 (Client) 的樣本數不等而導致評估出現偏差
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

    # 計算所有參與這輪測試的總樣本數
    total_examples = sum(num_examples for num_examples, _ in metrics)  
    if total_examples == 0: return {}  
    
    # 將各區的 [準確率/F1/AUC * 該區總樣本數] 加總後，除以所有區的總人數
    # 這能算出最真實且公平的全域表現
    aggregated_acc = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    aggregated_f1 = sum(num_examples * m.get("f1", 0.0) for num_examples, m in metrics) / total_examples
    aggregated_auc = sum(num_examples * m.get("auc", 0.0) for num_examples, m in metrics) / total_examples
    
    return {
        "accuracy": aggregated_acc,
        "f1": aggregated_f1,
        "auc": aggregated_auc
    }

# ==========================================
# 區塊 2：客製化聚合與存檔策略 (繼承自標準的 FedAvg)
# ==========================================
# 目的：攔截聚合過程，加入排序防呆機制，並將結果存入硬碟
class SaveModelStrategy(fl.server.strategy.FedAvg):
    
    # 建立時接收 model_path，決定這一輪實驗的結果要存在哪個資料夾
    def __init__(self, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.client_metrics_history = []  # 用來儲存各客戶端每輪的歷史紀錄清單
    
    
    # 攔截 fit (訓練) 階段
    def aggregate_fit(self, server_round, results, failures):
       
        # 【關鍵防呆】：聚合前，先依客戶端 id (cid) 強制排序結果。
        # 這是為了消除浮點數加總時的「順序誤差」(例如 A+B+C 與 C+B+A 在底層會有極微小差異)，確保實驗 100% 可重現。
        results.sort(key=lambda x: str(x[1].metrics.get("cid", "0")))
        
        # 呼叫原本的 FedAvg 邏輯，把各 client 訓練完的權重加總成新的 global model
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # 將最新的全域模型攔截並存到硬碟
        if aggregated_parameters is not None:
            # 將 Flower 的參數格式轉換成一般的 NumPy 陣列
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # 以 .npz 壓縮檔格式，獨立打包並覆寫儲存全域模型權重
            np.savez(self.model_path, *ndarrays)
            
        # 將 FedAvg 結果回傳給系統，準備發給各 client 進行下一輪
        return aggregated_parameters, aggregated_metrics


    # 攔截 evaluate (評估) 階段
    def aggregate_evaluate(self, server_round, results, failures):
        
        # 一樣在聚合評估前，先依客戶端 id 排序結果
        results.sort(key=lambda x: str(x[1].metrics.get("cid", "0")))
        
        # 將各客戶端回傳的評估指標抽出來，一筆一筆加入歷史紀錄清單中
        for client_proxy, eval_res in results:
            self.client_metrics_history.append({
                "server_round": server_round,
                "cid": eval_res.metrics.get("cid", "unknown"),
                "loss": eval_res.loss,
                "accuracy": eval_res.metrics.get("accuracy", 0.0),
                "f1": eval_res.metrics.get("f1", 0.0),
                "auc": eval_res.metrics.get("auc", 0.0)
            })
            
        # 每次 evaluate 結束就把歷史紀錄清單直接存成 CSV 檔
        # 這樣就算跑到一半斷線，之前的紀錄也都有保留
        history_path = self.model_path.replace(".npz", "_history.csv")
        pd.DataFrame(self.client_metrics_history).to_csv(history_path, index=False)
        
        return super().aggregate_evaluate(server_round, results, failures)


# ==========================================
# 區塊 3：策略打包與發號施令
# ==========================================
# 目的：建立第一代的空模型，並設定要傳給客戶端的參數
def get_strategy(seed: int, input_dim: int, local_epochs: int, lr: float, model_path: str):
    
    # 1. 鎖定伺服器的隨機種子：確保第一代發派出去的考卷 (模型權重)，不管跑幾次都長得一模一樣
    set_global_seed(seed)
    
    # 2. 建立第一代初始空模型
    initial_model = DiabetesNet(input_dim) 
    
    # 3. 把這個空模型的權重抽出來，轉換為 NumPy 陣列
    initial_parameters = [val.cpu().numpy() for val in initial_model.state_dict().values()]
    
    # 4. 定義 fit_config 閉包函式
    # 伺服器會透過這個函式，在每一輪開始時，把 local_epochs、lr 等指令派發給所有客戶端
    def fit_config(server_round: int):
        return {
            "local_epochs": local_epochs,
            "lr": lr,
            "server_round": server_round,
            "seed": seed  # 傳送 seed 讓 client 能夠據此推算並重設該回合的 random seed
        }
    
    # 5. 實例化我們剛剛寫好的 SaveModelStrategy
    strategy = SaveModelStrategy(
            model_path=model_path,
            fraction_fit=1.0,           # 每一輪都要挑選 100% 的 client 進行訓練
            fraction_evaluate=1.0,      # 每一輪都要挑選 100% 的 client 進行評估
            min_fit_clients=5,          # 必須收回 5 個 client 的權重後，才能進行 FedAvg 聚合
            min_evaluate_clients=5,     # 評估階段也必須要有 5 個 client 參與才算數
            min_available_clients=5,    # 伺服器啟動時，要有 5 個 client 連線才開始跑
            evaluate_metrics_aggregation_fn=weighted_average,  # 指定區塊 1 的加權平均函式
            on_fit_config_fn=fit_config,                       # 指定區塊 3 的指令發送函式
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters), # 放入第一代的初始權重
    )
    
    return strategy