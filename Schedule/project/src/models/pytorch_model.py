import torch
import torch.nn as nn

class DiabetesNet(nn.Module):
    def __init__(self, input_dim: int):
        super(DiabetesNet, self).__init__()
       
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)