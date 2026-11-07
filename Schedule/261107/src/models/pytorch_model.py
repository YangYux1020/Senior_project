# src/models/pytorch_model.py
import torch  
import torch.nn as nn  

class DiabetesNet(nn.Module):  
    def __init__(self, input_dim: int):  # input_dim: 疾病特徵數量
        super(DiabetesNet, self).__init__()  
        self.network = nn.Sequential(  # 建立序列容器
            
            nn.Linear(input_dim, 64),  # 第一層：輸入特徵 into 64個神經元
            nn.BatchNorm1d(64),  # 正規化
            nn.ReLU(),  
            nn.Dropout(0.2),  #隨機關閉20%神經元以防過擬合
            
            nn.Linear(64, 32),  # 第二層 過濾剩32個神經元
            nn.BatchNorm1d(32),  # 正規化
            nn.ReLU(),  
            nn.Dropout(0.2),  
            
            nn.Linear(32, 1)  # 輸出層：32 -> 單一輸出(原始預測分數)
        )

    #for train_utils.py --> outputs = model(inputs)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  #標示只有Tensor格式才能運算/回傳
        return self.network(x)  #當資料(X)進入，丟進上面的network運算