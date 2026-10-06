# src/train_utils.py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train(model: nn.Module, train_loader, epochs: int = 1, lr: float = 0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            
            # 使用 .view(-1) 對齊維度
            loss = criterion(outputs.view(-1), target.float())
            loss.backward()
            optimizer.step()
            
    return model

def evaluate(model: nn.Module, test_loader):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs.view(-1), target.float())
            total_loss += loss.item() * data.size(0)
            
            probs = torch.sigmoid(outputs.view(-1))
            preds = (probs >= 0.5).int()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    
    # 計算 F1 分數 (zero_division=0 避免無預測正例時報錯)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # 計算 ROC-AUC
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5  # 避免只有單一類別時計算報錯
        
    # 【關鍵修復】回傳包含 accuracy、f1、auc 的字典
    return float(avg_loss), {"accuracy": float(acc), "f1": float(f1), "auc": float(auc)}