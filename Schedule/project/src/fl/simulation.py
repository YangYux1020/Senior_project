import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import flwr as fl
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fl.client import DiabetesClient
from src.fl.server import strategy
from src.models.pytorch_model import DiabetesNet


def build_client_assets(cid: str, batch_size: int = 32):
    group_idx = int(cid) + 1
    train_path = f"./data/processed/partition_Group{group_idx}_train.parquet"
    test_path = f"./data/processed/partition_Group{group_idx}_test.parquet"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到分區資料檔: {train_path} 或 {test_path}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=["DIQ010"]).values
    y_train = train_df["DIQ010"].values

    X_test = test_df.drop(columns=["DIQ010"]).values
    y_test = test_df["DIQ010"].values

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = DiabetesNet(input_dim=input_dim)

    return model, train_loader, test_loader


def client_fn(cid):
    model, train_loader, test_loader = build_client_assets(cid)
    return DiabetesClient(model, train_loader, test_loader).to_client()


if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

# === 將以下程式碼接在 start_simulation(...) 之後 ===
    
    print("\n" + "="*60)
    print("開始步驟 7: 評估結果並更新本地模型 (三方比較)")
    print("="*60)
    import numpy as np
    from src.train_utils import train, evaluate
    
    partitions = ["0", "1", "2", "3", "4"]
    
    # 準備存放各客戶端資料的字典 (對應你的 train_loader[cid])
    train_loader = {}
    test_loader = {}
    for cid in partitions:
        _, tr_loader, te_loader = build_client_assets(cid)
        train_loader[cid] = tr_loader
        test_loader[cid] = te_loader

    # 1. 取得最終全域權重 (對應 .npz 讀取方式)
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../global_model.npz"))
    loaded_data = np.load(MODEL_PATH)
    # 將 npz 內的陣列依序提煉回 List 格式
    final_ndarrays = [loaded_data[f] for f in loaded_data.files]

    
    print(f"{'Client':<8} | {'Model Type':<18} | {'Accuracy':<8} | {'F1 Score':<8} | {'ROC-AUC':<8}")
    print("-" * 65)

    for cid in partitions:
        input_dim = train_loader[cid].dataset.tensors[0].shape[1]
        
        # 轉換全域權重格式供載入
        dummy_model = DiabetesNet(input_dim)
        params_dict = zip(dummy_model.state_dict().keys(), final_ndarrays)
        final_params = {k: torch.tensor(v) for k, v in params_dict}

        # [比較 1] 原始本地模型 (Local Only - 未經 FL)
        model_local = DiabetesNet(input_dim)
        model_local = train(model_local, train_loader[cid], epochs=3, lr=0.005)
        _, metrics_loc = evaluate(model_local, test_loader[cid])
        
        # [比較 2] 僅使用全域模型 (Global Only - 未經微調)
        model_global = DiabetesNet(input_dim)
        model_global.load_state_dict(final_params)
        _, metrics_glob = evaluate(model_global, test_loader[cid])
        
        # [比較 3] 個人化模型 (Personalized FL) - 完全套用你的基底程式碼
        model = DiabetesNet(input_dim)
        model.load_state_dict(final_params)
        model = train(model, train_loader[cid], epochs=3, lr=0.005) # 本地微調
        _, metrics_pers = evaluate(model, test_loader[cid])
        
        # 輸出該分區的比較結果
        print(f"Group {int(cid)+1} | {'Local Only':<18} | {metrics_loc['accuracy']:.4f}   | {metrics_loc['f1']:.4f}   | {metrics_loc['auc']:.4f}")
        print(f"Group {int(cid)+1} | {'Global Only':<18} | {metrics_glob['accuracy']:.4f}   | {metrics_glob['f1']:.4f}   | {metrics_glob['auc']:.4f}")
        print(f"Group {int(cid)+1} | {'Personalized FL':<18} | {metrics_pers['accuracy']:.4f}   | {metrics_pers['f1']:.4f}   | {metrics_pers['auc']:.4f}")
        print("-" * 65)