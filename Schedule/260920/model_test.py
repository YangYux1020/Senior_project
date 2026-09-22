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
# 進行模組功能整合測試
# ==========================================
if __name__ == "__main__":
    # 2. 設定硬體
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的運算硬體: {device}")
    
    # 3. 載入 Group1 的資料進行測試
    test_group = "Group1"
    train_loader, test_loader, input_dim = load_local_parquet_data(test_group)
    print(f"成功載入 {test_group} 資料！特徵維度: {input_dim}")
    
    # 4. 初始化模型
    model = DiabetesNet(input_dim=input_dim).to(device)
    
    # 5. 呼叫 train_utils.py 的 train 函式
    print("\n--- 開始模組化訓練 ---")
    trained_model = train(model, train_loader, epochs=30, lr=0.001, device=device)
    print("訓練完成！")
    
    # 6. 呼叫 train_utils.py 的 evaluate 函式
    print("\n--- 開始模組化評估 ---")
    loss, metrics = evaluate(trained_model, test_loader, device=device)
    
    print(f"測試集 Loss: {loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")