#src/train_utils.py

import torch  
import torch.nn as nn  
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model: nn.Module, train_loader, epochs: int = 1, lr: float = 0.001):
    criterion = nn.BCEWithLogitsLoss()  #loss function, 把DiabetesNet的原始分數轉換成0-1間的罹病機率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 設定優化器與學習率
    
    model = model.to(device) # 確保模型本身被搬移到 GPU
    model.train()  #訓練模式
    for _ in range(epochs):  
        for data, target in train_loader:  
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 清空前一次的梯度
            outputs = model(data)  # foward得到原始預測分數
            
            loss = criterion(outputs.view(-1), target.float())
            loss.backward()  # 反向傳播
            optimizer.step()  # 根據梯度更新模型權重
            
    return model  # 回傳新模型給client.py


def evaluate(model: nn.Module, test_loader):
    criterion = nn.BCEWithLogitsLoss()  # 使用相同的loss function
    model = model.to(device) # 確保模型本身被搬移到 GPU
    model.eval()  #評估模式(關閉 Dropout)
    
    total_loss = 0.0  # 初始化總誤差
    all_preds, all_targets, all_probs = [], [], []  # list for(判定/標籤/預設機率)
    
    with torch.no_grad():  # 關閉梯度計算，節省記憶體
        for data, target in test_loader:
            
            data, target = data.to(device), target.to(device) #將驗證資料與標籤搬移到 GPU
              
            outputs = model(data)  
            loss = criterion(outputs.view(-1), target.float())  
            total_loss += loss.item() * data.size(0)  # 還原並累加該批次的絕對誤差
            
            probs = torch.sigmoid(outputs.view(-1))  # 轉換為0-1的機率
            preds = (probs >= 0.5).int()  #0.5為門檻判斷為罹病(1)或沒病(0)
            
            # 將tensor轉為NumPy陣列並收集
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    avg_loss = total_loss / len(test_loader.dataset)  #計算測試集的平均誤差
    acc = accuracy_score(all_targets, all_preds)  # 計算Accuracy
    
    
    # 計算F1 score (zero_division=0 為了避免所有病患都判為沒病(0))
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    
    # 計算ROC-AUC
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5  # 若測試集全為健康/罹病，會無法判斷，就給 0.5 基準分避免崩潰
        
    # 把所有np或pytorch的格式轉換成flower認得的float格式
    return float(avg_loss), {"accuracy": float(acc), "f1": float(f1), "auc": float(auc)}

