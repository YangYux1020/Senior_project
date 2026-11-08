import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.models.pytorch_model import DiabetesNet


SPLIT_SEED = 9527


def build_client_assets(cid: str, batch_size: int = 32, sample_ratio: float = 1.0, subsample_seed: int = 9527):
    # 照client id(cid) 載入對應的Group資料
    group_idx = int(cid) + 1
    train_path = f"./data/processed/partition_Group{group_idx}_train.parquet"
    test_path = f"./data/processed/partition_Group{group_idx}_test.parquet"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到分區資料檔：{train_path} 或 {test_path}")

    # 讀資料並剃除DIQ010標籤
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train_full = train_df.drop(columns=["DIQ010"]).values
    y_train_full = train_df["DIQ010"].values
    X_test = test_df.drop(columns=["DIQ010"]).values
    y_test = test_df["DIQ010"].values

    # 切割訓練集 (85%) 與 驗證集 (15%)
    # 加入 stratify=y_train_full 確保切分後的糖尿病患比例與原本一致
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15,
        random_state=SPLIT_SEED,
        stratify=y_train_full
    )

    # 支援小醫院少樣本標註稀缺實驗 (Few-Shot / Scarcity Regime)
    if 0.0 < sample_ratio < 1.0:
        n_sub = max(16, int(round(len(X_train) * sample_ratio)))
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=n_sub,
            random_state=subsample_seed,
            stratify=y_train
        )

    # 將 NumPy 陣列轉為 PyTorch tensor並綁定為 Dataset
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    # 新增驗證集 Dataset
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    # 替訓練集配有種子的Generator，確保每一輪都是同樣順序
    g = torch.Generator()

    eff_bs = min(batch_size, max(8, len(X_train) // 2))
    train_loader = DataLoader(train_dataset, batch_size=eff_bs, shuffle=True, generator=g)  # 訓練集
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)              # 驗證集
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)            # 測試集

    # 實例化神經模型
    input_dim = X_train.shape[1]  # 特徵數目
    model = DiabetesNet(input_dim=input_dim)

    return model, train_loader, val_loader, test_loader
