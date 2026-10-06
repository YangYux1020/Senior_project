# src/fl/server.py
import os
import flwr as fl
import numpy as np
from flwr.common import Metrics
from typing import List, Tuple

# 改為 .npz 格式
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../global_model.npz"))

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0: return {}
    aggregated_acc = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    return {"accuracy": aggregated_acc}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # 使用 np.savez 支援儲存多個不同形狀的權重矩陣
            np.savez(MODEL_PATH, *ndarrays)
        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    evaluate_metrics_aggregation_fn=weighted_average,
)