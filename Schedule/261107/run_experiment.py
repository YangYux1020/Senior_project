# run_experiment.py

import os
import sys
import json
import argparse
import platform
import pandas as pd
import torch
import flwr as fl

# 將 src 目錄加入系統路徑，確保能順利讀取專案內的模組
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.seed import set_global_seed
from src.fl.data import build_client_assets
from src.fl.client import DiabetesClient
from src.fl.server import get_strategy

# ==========================================
# 區塊 1：定義客戶端產生器 (Client Factory)
# ==========================================
# 說明：Flower 模擬器在啟動時，會一直呼叫這個函式來產生 5 個虛擬客戶端
def client_fn(cid: str):
    # 使用我們在 data.py 寫好的函式，載入對應 cid (例如 "0"~"4") 的資料
    # 注意：這裡刻意不接收 test_loader，將其封存不用
    model, train_loader, val_loader, _ = build_client_assets(cid)
    
    # 將 cid、模型、訓練集與「驗證集」綁定到客戶端，回傳給 Flower 框架
    return DiabetesClient(cid, model, train_loader, val_loader).to_client()


if __name__ == "__main__":
    # ==========================================
    # 區塊 2：接收外部指令與建立專屬資料夾
    # ==========================================
    # 說明：讓這支程式能透過終端機指令動態改變參數 (例如 python run_experiment.py --seed 0 --lr 0.01)
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Runner")
    parser.add_argument("--seed", type=int, required=True, help="執行種子 (控制隨機性)")
    parser.add_argument("--local_epochs", type=int, required=True, help="本地訓練回合數")
    parser.add_argument("--lr", type=float, required=True, help="學習率")
    parser.add_argument("--rounds", type=int, required=True, help="聯邦學習總輪數")
    parser.add_argument("--output_folder", type=str, required=True, help="輸出資料夾路徑")
    args = parser.parse_args()

    # 為本次實驗建立專屬的輸出資料夾 (若不存在則建立)
    os.makedirs(args.output_folder, exist_ok=True)

    # ==========================================
    # 區塊 3：環境設定與啟動模擬
    # ==========================================
    # 1. 呼叫我們寫好的 seed.py，鎖定整個實驗的基礎起跑線
    set_global_seed(args.seed)

    # 2. 小技巧：因為模型需要 input_dim，我們先偷拿第 0 號客戶端的資料來探測特徵數量
    _, tr_loader, _, _ = build_client_assets("0")
    input_dim = tr_loader.dataset.tensors[0].shape[1]

    # 設定全局模型權重要存放在這次實驗的專屬資料夾內
    model_path = os.path.join(args.output_folder, "global_model.npz")

    # 3. 呼叫 server.py 的 get_strategy，建立伺服器策略與初始模型
    strategy = get_strategy(
        seed=args.seed, 
        input_dim=input_dim, 
        local_epochs=args.local_epochs, 
        lr=args.lr, 
        model_path=model_path
    )

    # 設定每個虛擬客戶端可使用的硬體資源 (5 個客戶端共享 1 張 GPU，各分得 0.2)
    # 限制每個客戶端的資源，並限制 Ray 總共只能動用 6 個 CPU 核心
    client_resources = {
        "num_cpus": 2,
        "num_gpus": 0.2 if torch.cuda.is_available() else 0.0
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"num_cpus": 6}  # 新增這行，防止 Ray 把 16 個 CPU 核心全部吃滿
    )

    # ==========================================
    # 區塊 4：戰果結算與存檔
    # ==========================================
    # 紀錄一：各客戶端的體檢表 (包含各 Client 每輪的 Loss, AUC, F1)
    client_history_path = os.path.join(args.output_folder, "client_metrics.csv")
    pd.DataFrame(strategy.client_metrics_history).to_csv(client_history_path, index=False)

    # 紀錄二：實驗設定檔 (留下本次實驗的參數與環境版本，方便日後除錯與追溯)
    config_data = {
        "seed": args.seed,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "rounds": args.rounds,
        "output_folder": args.output_folder,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "flwr_version": fl.__version__,
    }
    with open(os.path.join(args.output_folder, "config.json"), "w") as f:
        json.dump(config_data, f, indent=4)

    # 紀錄三：從 Flower 回傳的 history 物件中，萃取全域 (Global) 的歷史表現
    global_history_list = []
    if history.metrics_distributed:
        losses = history.losses_distributed
        accuracies = history.metrics_distributed.get("accuracy", [])
        f1_scores = history.metrics_distributed.get("f1", [])
        aucs = history.metrics_distributed.get("auc", [])
        
        # 整理每一輪的成效數據
        for i in range(len(losses)):
            round_num = losses[i][0]
            global_history_list.append({
                "round": round_num,
                "val_loss": losses[i][1] if i < len(losses) else None,
                "accuracy": accuracies[i][1] if i < len(accuracies) else None,
                "f1": f1_scores[i][1] if i < len(f1_scores) else None,
                "auc": aucs[i][1] if i < len(aucs) else None
            })

    # 【神來一筆】：把 global history 放在所有程式的最最最後面才寫入硬碟
    # 這可以當作一個「旗標」：如果資料夾裡有這個檔案，代表這場實驗完美順利跑完，沒有跑到一半當機
    global_history_path = os.path.join(args.output_folder, "global_history.csv")
    pd.DataFrame(global_history_list).to_csv(global_history_path, index=False)
    
    print(f"\n 實驗執行完畢！所有紀錄已完整存入: {args.output_folder}")