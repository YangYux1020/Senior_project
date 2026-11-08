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
    def __init__(self, cid: str, model: DiabetesNet, train_loader, val_loader):
        self.cid = int(cid)  # 將 id 以整數保存，方便後續伺服器排序
        self.model = model   # 綁定本地端的神經網路模型
        self.train_loader = train_loader
        self.val_loader = val_loader  # 嚴格遵守指南：調參期間只用驗證集 (val)，不用測試集 (test)

    # ==========================================
    # 區塊 2：權重翻譯官 (PyTorch Tensor <-> NumPy Array)
    # ==========================================
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    # ==========================================
    # 區塊 3：接收指令與在地訓練 (支援 FedAvg 與 FedProx)
    # ==========================================
    def fit(self, parameters, config):
        server_round = config["server_round"]
        base_seed = config["seed"]
        call_seed = base_seed + 1000 * server_round + self.cid

        set_global_seed(call_seed)
        self.train_loader.generator.manual_seed(call_seed)
        self.set_parameters(parameters)

        epochs = config["local_epochs"]
        lr = config["lr"]
        proximal_mu = float(config.get("proximal_mu", 0.0))

        self.model = train(
            self.model,
            self.train_loader,
            epochs=epochs,
            lr=lr,
            proximal_mu=proximal_mu,
        )

        return self.get_parameters(config={}), len(self.train_loader.dataset), {"cid": self.cid}

    # ==========================================
    # 區塊 4：模擬考與回報成績
    # ==========================================
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = evaluate(self.model, self.val_loader)
        metrics_float = {k: float(v) for k, v in metrics.items()}
        metrics_float["cid"] = self.cid
        return float(loss), len(self.val_loader.dataset), metrics_float
