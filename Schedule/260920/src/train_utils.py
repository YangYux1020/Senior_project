import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def train(model, dataloader, epochs, lr, device='cpu'):
    """
    訓練神經網路模型的核心函式。
    
    參數:
        model: 要訓練的 PyTorch 模型 (對應 DiabetesNet)
        dataloader: 提供訓練批次數據的 DataLoader
        epochs: 訓練的總輪數
        lr: 學習率 (Learning Rate)
        device: 執行運算的硬體 ('cpu' 或 'cuda')
        
    回傳:
        model: 訓練完成的模型
    """
    # 因 DiabetesNet 輸出為 1 維且有 Sigmoid，必須用二元交叉熵損失函數 (衡量模型預測的機率分佈與真實標籤（0 或 1）之間的差距)
    criterion = nn.BCELoss() 
    
    # 使用 Adam 優化器自動調整學習率並更新權重
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 宣告模型進入訓練模式 (讓系統準備好計算梯度)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            # 將特徵搬移至指定硬體
            batch_X = batch_X.to(device)
            
            # 且形狀必須與模型輸出一致 (batch_size, 1)，因此需使用 view(-1, 1) 擴充維度
            batch_y = batch_y.to(device).float().view(-1, 1)
            """
            擴充維度的目的： 
            如果直接將二維的預測值 [32, 1] 與一維的真實標籤 [32] 丟給 BCELoss
            PyTorch 會因為兩者維度不匹配而無法正確計算誤差
            甚至觸發廣播機制(Broadcasting)導致嚴重的邏輯錯誤
            透過擴充維度(.unsqueeze(1))或 .view(-1, 1))
            可以強制將一維陣列轉換為二維的直行向量 [32, 1]
            確保預測機率與真實答案在數學維度上達到 1:1 的完美對齊。
            """

            
            optimizer.zero_grad() # 1. 清空上一輪的梯度
            outputs = model(batch_X) # 2. 前向傳播：讓模型給出預測機率 (0~1 之間)  
            loss = criterion(outputs, batch_y) # 3. 裁判計算損失 (Loss)
            loss.backward() # 4. 反向傳播：計算微積分梯度
            optimizer.step() # 5. 教練根據梯度更新權重

            epoch_loss += loss.item()
            
        # 每 10 個 Epoch 印出一次訓練進度
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], 平均 Loss: {avg_loss:.4f}")
            
            
    return model


def evaluate(model, dataloader, device='cpu'):
    """
    評估模型效能的函式，包含混淆矩陣與各項指標。
    """
    criterion = nn.BCELoss()
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).float().view(-1, 1)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            predicted = (outputs >= 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        roc_auc = 0.0 
        
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }
    
    return avg_loss, metrics