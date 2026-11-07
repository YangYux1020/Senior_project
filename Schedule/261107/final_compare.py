# final_compare.py
import os
import sys
import numpy as np
import pandas as pd
import torch

from src.seed import set_global_seed
from src.fl.data import build_client_assets
from src.models.pytorch_model import DiabetesNet
from src.train_utils import train, evaluate, device

def load_global_model_weights(model: torch.nn.Module, npz_path: str):
    """載入保存在 .npz 檔案中的聯邦全域模型權重並寫入 PyTorch 模型"""
    data = np.load(npz_path)
    weights = [data[f] for f in data.files]
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model

def run_final_comparison():
    final_seeds = [10, 11, 12, 13, 14]
    client_cids = ["0", "1", "2", "3", "4"]
    
    # Stage A 與 Stage B 決策的最佳參數
    best_E = 2
    best_lr = 0.01
    num_rounds = 20
    total_local_epochs = num_rounds * best_E  # 20 * 2 = 40 個週期，保證訓練總量公平
    
    # 個人化微調設定
    pers_epochs = 3
    pers_lr = 0.005

    records = []
    output_dir = "results/final"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print(f"開始執行測試集最終對比評估 (使用裝置: {device})")
    print(f"評估設定: E*={best_E}, lr*={best_lr} | 測試種子: {final_seeds}")
    print("=" * 75)

    for seed in final_seeds:
        npz_path = os.path.join(output_dir, f"E{best_E}_lr{best_lr}", f"seed{seed}", "global_model.npz")
        if not os.path.exists(npz_path):
            print(f"[錯誤] 找不到全域模型權重檔案: {npz_path}")
            print("請先確認是否已執行過: python sweep.py --stage final")
            sys.exit(1)

        for idx, cid in enumerate(client_cids):
            group_name = f"Group{int(cid) + 1}"
            
            # 計算專屬於該種子與客戶端的確定性種子
            client_seed = seed * 100 + idx
            set_global_seed(client_seed)
            
            # 正確接收 build_client_assets 的 4 個回傳值 (model, train_loader, val_loader, test_loader)
            _, train_loader, val_loader, test_loader = build_client_assets(cid)
            
            # 從 DataLoader 動態獲取特徵維度 (NHANES 為 13 維)
            input_dim = next(iter(train_loader))[0].shape[1]

            # ---------------------------------------------------------
            # 模型一：僅本地模型 (Local-only)
            # ---------------------------------------------------------
            set_global_seed(client_seed)
            if hasattr(train_loader, 'generator') and train_loader.generator is not None:
                train_loader.generator.manual_seed(client_seed)
                
            local_model = DiabetesNet(input_dim=input_dim).to(device)
            local_model = train(local_model, train_loader, epochs=total_local_epochs, lr=best_lr)
            local_loss, local_metrics = evaluate(local_model, test_loader)
            
            records.append({
                "seed": seed,
                "client": group_name,
                "model_type": "Local-only",
                "loss": local_loss,
                "accuracy": local_metrics["accuracy"],
                "f1": local_metrics["f1"],
                "auc": local_metrics["auc"]
            })

            # ---------------------------------------------------------
            # 模型二：僅全域模型 (Global-only)
            # ---------------------------------------------------------
            global_model = DiabetesNet(input_dim=input_dim).to(device)
            global_model = load_global_model_weights(global_model, npz_path)
            global_loss, global_metrics = evaluate(global_model, test_loader)

            records.append({
                "seed": seed,
                "client": group_name,
                "model_type": "Global-only",
                "loss": global_loss,
                "accuracy": global_metrics["accuracy"],
                "f1": global_metrics["f1"],
                "auc": global_metrics["auc"]
            })

            # ---------------------------------------------------------
            # 模型三：個人化微調模型 (Personalized)
            # ---------------------------------------------------------
            set_global_seed(client_seed)
            if hasattr(train_loader, 'generator') and train_loader.generator is not None:
                train_loader.generator.manual_seed(client_seed)
                
            pers_model = DiabetesNet(input_dim=input_dim).to(device)
            pers_model = load_global_model_weights(pers_model, npz_path)
            pers_model = train(pers_model, train_loader, epochs=pers_epochs, lr=pers_lr)
            pers_loss, pers_metrics = evaluate(pers_model, test_loader)

            records.append({
                "seed": seed,
                "client": group_name,
                "model_type": "Personalized",
                "loss": pers_loss,
                "accuracy": pers_metrics["accuracy"],
                "f1": pers_metrics["f1"],
                "auc": pers_metrics["auc"]
            })

        print(f"已完成 [種子 {seed}] 下所有客戶端 (Group1 ~ Group5) 的三方模型評估。")

    # 儲存明細表
    df_results = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "final_test_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n所有評估明細已存入: {csv_path}")

    # 表一：各分區跨 5 個種子的平均值與標準差
    print("\n" + "=" * 85)
    print("表一：各分區測試集最終表現 (跨 5 個種子之 平均值 ± 標準差)")
    print("=" * 85)

    summary_client = df_results.groupby(["client", "model_type"], sort=False)[["auc", "f1", "accuracy", "loss"]].agg(
        lambda x: f"{np.mean(x):.4f} ± {np.std(x):.4f}"
    ).reset_index()
    print(summary_client.to_string(index=False))

    # 表二：跨所有分區與種子的綜合整體平均
    print("\n" + "=" * 85)
    print("表二：三種模型架構之整體平均表現 (Overall Mean ± Std across all Clients & Seeds)")
    print("=" * 85)

    summary_overall = df_results.groupby(["model_type"], sort=False)[["auc", "f1", "accuracy", "loss"]].agg(
        lambda x: f"{np.mean(x):.4f} ± {np.std(x):.4f}"
    ).reset_index()
    print(summary_overall.to_string(index=False))

if __name__ == "__main__":
    run_final_comparison()