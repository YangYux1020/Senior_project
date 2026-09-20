import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path, scaler=None, fit=False):
    """
    載入並預處理 NHANES 分區資料。
    
    參數:
        path (str): CSV 檔案的相對路徑。
        scaler (StandardScaler, optional): 用於特徵縮放的物件。
        fit (bool, optional): 是否要在這份資料上擬合 scaler。
        
    回傳:
        X_train_df (pd.DataFrame): 訓練集 (特徵與標籤合併)
        X_test_df (pd.DataFrame): 測試集 (特徵與標籤合併)
        scaler (StandardScaler): 處理過後的縮放器
    """
    df = pd.read_csv(path)
    
    # 2. 標籤轉換與噪音清除
    df = df[df['DIQ010'] != 9.0].reset_index(drop=True)
    df['DIQ010'] = df['DIQ010'].replace({3.0: 1.0, 2.0: 0.0})
    
    # 3. 特徵工程
    df['GH_GLU_Interact'] = df['LBXGH'] * df['LBXGLU']
    df['UACR_Proxy'] = df['URXUMA'] / df['LBXSCR']
    df['Lipid_Ratio'] = df['LBDLDL'] / df['LBDHDD']
    
    # 4. 分離特徵與標籤
    X = df.drop(columns=['SEQN', 'DIQ010'], errors='ignore')
    Y = df['DIQ010']
    
    # 5. 切分資料
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    # 6. 特徵縮放 (直接使用 raw 資料)
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
    else:
        if fit:
            X_train_scaled = scaler.fit_transform(X_train_raw)
        else:
            X_train_scaled = scaler.transform(X_train_raw)

    X_test_scaled = scaler.transform(X_test_raw)

    # 7. 轉回 Pandas DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 8. 合併二元標籤
    X_train_df['DIQ010'] = Y_train.values
    X_test_df['DIQ010'] = Y_test.values
    

    return X_train_df, X_test_df, scaler