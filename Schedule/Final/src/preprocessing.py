import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path, scaler=None, fit=False):
    
    df = pd.read_csv(path)

    df = df[df['DIQ010'] != 9.0].reset_index(drop=True)
    df['DIQ010'] = df['DIQ010'].replace({3.0: 1.0, 2.0: 0.0})
    
    df['GH_GLU_Interact'] = df['LBXGH'] * df['LBXGLU']
    df['UACR_Proxy'] = df['URXUMA'] / df['LBXSCR']
    df['Lipid_Ratio'] = df['LBDLDL'] / df['LBDHDD']
    
   
    X = df.drop(columns=['SEQN', 'DIQ010'], errors='ignore')
    Y = df['DIQ010']
    
   
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
    else:
        if fit:
            X_train_scaled = scaler.fit_transform(X_train_raw)
        else:
            X_train_scaled = scaler.transform(X_train_raw)

    X_test_scaled = scaler.transform(X_test_raw)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    X_train_df['DIQ010'] = Y_train.values
    X_test_df['DIQ010'] = Y_test.values
    
    return X_train_df, X_test_df, scaler