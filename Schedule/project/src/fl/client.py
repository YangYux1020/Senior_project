import flwr as fl
import torch
from src.models.pytorch_model import DiabetesNet
from src.train_utils import train, evaluate

class DiabetesClient(fl.client.NumPyClient):
    def __init__(self, model: DiabetesNet, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        """將 PyTorch 權重張量轉為 NumPy 陣列傳送給伺服器"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """接收伺服器的權重更新，覆寫至本地模型中"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """執行本地訓練：載入全域權重 -> 跑本地 Epoch -> 回傳更新權重"""
        self.set_parameters(parameters)
        
        # 讀取傳入的超參數或使用預設值
        epochs = config.get("local_epochs", 1)
        lr = config.get("lr", 0.01)
        
        self.model = train(self.model, self.train_loader, epochs=epochs, lr=lr)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """評估模型表現：在本地測試集驗證損失與準確率/AUC"""
        self.set_parameters(parameters)
        loss, metrics = evaluate(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), metrics