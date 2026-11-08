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
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated_acc = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    aggregated_f1 = sum(num_examples * m.get("f1", 0.0) for num_examples, m in metrics) / total_examples
    aggregated_auc = sum(num_examples * m.get("auc", 0.0) for num_examples, m in metrics) / total_examples

    return {
        "accuracy": aggregated_acc,
        "f1": aggregated_f1,
        "auc": aggregated_auc,
    }


# ==========================================
# 區塊 2：客製化聚合與存檔策略 (繼承自標準的 FedAvg)
# ==========================================
class SaveModelStrategy(fl.server.strategy.FedAvg):

    def __init__(self, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.client_metrics_history = []

    def aggregate_fit(self, server_round, results, failures):
        results.sort(key=lambda x: int(x[1].metrics.get("cid", 0)))
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            np.savez(self.model_path, *ndarrays)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        results.sort(key=lambda x: int(x[1].metrics.get("cid", 0)))

        for client_proxy, eval_res in results:
            self.client_metrics_history.append({
                "server_round": server_round,
                "cid": eval_res.metrics.get("cid", "unknown"),
                "loss": eval_res.loss,
                "accuracy": eval_res.metrics.get("accuracy", 0.0),
                "f1": eval_res.metrics.get("f1", 0.0),
                "auc": eval_res.metrics.get("auc", 0.0),
            })

        history_path = self.model_path.replace(".npz", "_history.csv")
        pd.DataFrame(self.client_metrics_history).to_csv(history_path, index=False)

        return super().aggregate_evaluate(server_round, results, failures)


# ==========================================
# 區塊 3：策略打包與發號施令 (支援 FedAvg 與 FedProx proximal_mu)
# ==========================================
def get_strategy(seed: int, input_dim: int, local_epochs: int, lr: float, model_path: str, proximal_mu: float = 0.0):
    set_global_seed(seed)
    initial_model = DiabetesNet(input_dim)
    initial_parameters = [val.cpu().numpy() for val in initial_model.state_dict().values()]

    def fit_config(server_round: int):
        return {
            "local_epochs": local_epochs,
            "lr": lr,
            "proximal_mu": proximal_mu,
            "server_round": server_round,
            "seed": seed,
        }

    strategy = SaveModelStrategy(
        model_path=model_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
    )

    return strategy
