# run_experiment.py

import os
import sys
import json
import argparse
import platform
import pandas as pd
import torch
import flwr as fl

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.seed import set_global_seed
from src.fl.data import build_client_assets
from src.fl.client import DiabetesClient
from src.fl.server import get_strategy


def client_fn(cid: str):
    model, train_loader, val_loader, _ = build_client_assets(cid)
    return DiabetesClient(cid, model, train_loader, val_loader).to_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Runner")
    parser.add_argument("--seed", type=int, required=True, help="執行種子 (控制隨機性)")
    parser.add_argument("--local_epochs", type=int, required=True, help="本地訓練回合數")
    parser.add_argument("--lr", type=float, required=True, help="學習率")
    parser.add_argument("--rounds", type=int, required=True, help="聯邦學習總輪數")
    parser.add_argument("--output_folder", type=str, required=True, help="輸出資料夾路徑")
    parser.add_argument("--proximal_mu", type=float, default=0.0, help="FedProx 近端正則化係數 mu (0.0 代表標準 FedAvg)")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    set_global_seed(args.seed)

    _, tr_loader, _, _ = build_client_assets("0")
    input_dim = tr_loader.dataset.tensors[0].shape[1]

    model_path = os.path.join(args.output_folder, "global_model.npz")

    strategy = get_strategy(
        seed=args.seed,
        input_dim=input_dim,
        local_epochs=args.local_epochs,
        lr=args.lr,
        model_path=model_path,
        proximal_mu=args.proximal_mu,
    )

    client_resources = {
        "num_cpus": 2,
        "num_gpus": 0.2 if torch.cuda.is_available() else 0.0,
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"num_cpus": 6},
    )

    client_history_path = os.path.join(args.output_folder, "client_metrics.csv")
    pd.DataFrame(strategy.client_metrics_history).to_csv(client_history_path, index=False)

    config_data = {
        "seed": args.seed,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "proximal_mu": args.proximal_mu,
        "algorithm": "FedProx" if args.proximal_mu > 0 else "FedAvg",
        "rounds": args.rounds,
        "output_folder": args.output_folder,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "flwr_version": fl.__version__,
    }
    with open(os.path.join(args.output_folder, "config.json"), "w") as f:
        json.dump(config_data, f, indent=4)

    global_history_list = []
    if history.metrics_distributed:
        losses = history.losses_distributed
        accuracies = history.metrics_distributed.get("accuracy", [])
        f1_scores = history.metrics_distributed.get("f1", [])
        aucs = history.metrics_distributed.get("auc", [])

        for i in range(len(losses)):
            round_num = losses[i][0]
            global_history_list.append({
                "round": round_num,
                "val_loss": losses[i][1] if i < len(losses) else None,
                "accuracy": accuracies[i][1] if i < len(accuracies) else None,
                "f1": f1_scores[i][1] if i < len(f1_scores) else None,
                "auc": aucs[i][1] if i < len(aucs) else None,
            })

    global_history_path = os.path.join(args.output_folder, "global_history.csv")
    pd.DataFrame(global_history_list).to_csv(global_history_path, index=False)

    print(f"\n 實驗執行完畢！所有紀錄已完整存入: {args.output_folder}")
