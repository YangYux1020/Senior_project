import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path, scaler=None, imputer=None, fit=False):
    
    """
    載入並預處理 NHANES 分區資料，確保所有客戶端資料格式與縮放標準一致。
    
    參數:
        path (str): CSV 檔案的相對路徑。
        scaler (StandardScaler, optional): 用於特徵縮放的物件。若為 None，則函式內部會自動建立。
        fit (bool, optional): 是否要在這份資料上擬合 (fit) scaler。通常只有第一份基準資料設為 True，其餘分區設為 False。
        
    回傳:
        X_scaled (np.ndarray): 縮放後的特徵矩陣。
        Y (pd.Series): 標準化的二元標籤。
        scaler (StandardScaler): 處理過後的縮放器，供後續分區使用。
    """
    df = pd.read_csv(path)
    
    # 2. 標籤轉換與噪音清除 (二元分類標準化)
    df = df[df['DIQ010'] != 9.0].reset_index(drop=True) # 移除 9.0 (拒絕回答/不知道)
    df['DIQ010'] = df['DIQ010'].replace({3.0: 1.0, 2.0: 0.0}) # 將 3.0 (邊緣) 轉為 1.0 (有風險)，2.0 (健康) 轉為 0.0
    
    # 3. 特徵工程：建立衍生變數 (若資料集包含所需欄位)
    df['GH_GLU_Interact'] = df['LBXGH'] * df['LBXGLU']
    df['UACR_Proxy'] = df['URXUMA'] / df['LBXSCR']
    df['Lipid_Ratio'] = df['LBDLDL'] / df['LBDHDD']
    
    # 4. 分離特徵與標籤，並移除序列號
    if 'SEQN' in df.columns:
        X = df.drop(columns=['SEQN', 'DIQ010'])
    else:
        X = df.drop(columns=['DIQ010'])
    Y = df['DIQ010']
    
    # 5. 缺失值處理 (確保聯邦學習輸入穩定)
    imputer = SimpleImputer(strategy='median') # 使用中位數填補可能的 NaN
    X_imputed = imputer.fit_transform(X) # 找到 X 的整體統計特性之指標 (平均值、標準差、最大最小值等等) 再套用後面的 transform() 動作上實行
    
    # 6. 特徵縮放 (聯邦學習最重要的一步)
    if scaler is None:
        scaler = StandardScaler()
        
    if fit:
        X_scaled = scaler.fit_transform(X_imputed) # 僅在基準分區（或集中式訓練集）上計算均值與標準差
    else:
        X_scaled = scaler.transform(X_imputed) # 其他分區嚴格套用基準分區的縮放標準，避免資料外洩與尺度不一
        
    return X_scaled, Y, scaler 
    # X_scaled (np.ndarray): 縮放後的特徵矩陣。
    # Y (pd.Series): 標準化的二元標籤。
    # scaler (StandardScaler): 處理過後的縮放器，供後續分區使用。
 



def prepare_data_splits(X, Y, test_size=0.2, random_state=42, scaler=None):
    """
    針對交叉驗證 (CV) 設計的資料切分與縮放函式。
    僅切分訓練集與測試集，將驗證流程交由 CV 演算法內部執行。
    """
    # 第一步：單次切分出訓練集與測試集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, Y_train, Y_test
    