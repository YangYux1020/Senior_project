import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess(path, scaler=None, imputer=None, fit=False):
    df = pd.read_csv(path)
    # ... 所有分區使用相同的欄位選取／編碼方式 ...
    
    if fit:
    imputer = SimpleImputer(strategy='median').fit(df[num_cols])
    scaler = StandardScaler().fit(imputer.transform(df[num_cols]))
    df[num_cols] = scaler.transf