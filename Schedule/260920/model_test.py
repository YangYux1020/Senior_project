import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. 從 src 資料夾匯入您剛寫好的兩個模組
from src.models.pytorch_model import DiabetesNet
from src.train_utils import train, evaluate

def load_local_parquet_data(group_name, batch_size=32):
    """讀取處理好的 Parquet 檔案並轉換為 DataLoader"""
    # 假設 Notebook 位於專案根目錄
    current_dir = os.getcwd()
    train_path = os.path.join(current_dir, 'data', 'processed', 'train', f'partition_{group_name}_train.parquet')
    test_path = os.path.join(current_dir, 'data', 'processed', 'test', f'partition_{group_name}_test.parquet')
    
    # 讀取 Parquet
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # 分離特徵與標籤
    X_train = train_df.drop(columns=['DIQ010']).values
    Y_train = train_df['DIQ010'].values
    
    X_test = test_df.drop(columns=['DIQ010']).values
    Y_test = test_df['DIQ010'].values
    
    # 轉換為 PyTorch Tensor，並針對 BCELoss 將 Y 擴充維度 (unsqueeze)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)
    
    # 封裝為 DataLoader
    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_train.shape[1]

# ==========================================
# 進行模組功能整合測試 (循序訓練版本)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的運算硬體: {device}")
    
    test_group = ["Group1", "Group2", "Group3", "Group4", "Group5"]
    
    # 1. 取得特徵維度並在迴圈「外部」初始化模型
    # 確保這是一個會不斷累積經驗的單一實體，不會被覆蓋
    _, _, input_dim = load_local_parquet_data("Group1")
    model = DiabetesNet(input_dim=input_dim).to(device)
    print(f"\n=== 全局模型初始化完成 (特徵維度: {input_dim}) ===")
    
    # 2. 階段一：循序訓練 (Sequential Training)
    for group in test_group:
        train_loader, _, _ = load_local_parquet_data(group)
    
        print(f"\n--- 正在使用 {group} 訓練集更新模型權重 ---")
        # 傳入同一個 model 變數，它會保留上一組學到的權重繼續訓練
        # 由於是 5 組資料接力訓練，建議將單組 epoch 調降 (例如 10~20)，避免過度擬合最後一組
        model = train(model, train_loader, epochs=50, lr=0.001, device=device)
        print(f"{group} 訓練階段完成！")

    # 3. 階段二：最終綜合評比 (Final Evaluation)
    print("\n==========================================")
    print("=== 最終模型綜合評估 (Final Evaluation) ===")
    print("==========================================")
    
    # 讓吸收了 5 組資料經驗的最終模型，依序挑戰 5 個測試集
    for group in test_group:
        _, test_loader, _ = load_local_parquet_data(group)
        loss, metrics = evaluate(model, test_loader, device=device)
        
        print(f"\n挑戰 {group} 測試集:")
        print(f"Loss: {loss:.4f} | Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")