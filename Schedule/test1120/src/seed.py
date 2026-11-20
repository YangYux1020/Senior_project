import os
import random
import numpy as np
import torch

def set_global_seed(seed: int):
    """
    【主要連動檔案：run_experiment.py, check_repro.py, final_compare.py】
    說明：這三個主程式必須在開頭第一行呼叫此函式，為整個腳本定下基準種子。
    """
    
    # 1. 鎖定 Python 內建環境的隨機性
    # 若未設定此變數（或設為 random），Python 會在每次啟動時於 [0, 4294967295] 範圍內隨機產生一個數值來作為雜湊種子。
    # 【連動檔案：client.py (平行的 Ray 工作程序)】
    # 說明：在客戶端程式碼內 (如每次 fit 開始時) 需再次呼叫，因為主程式開頭的種子不會傳到 Ray 的獨立程序中。
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    # 2. 鎖定 NumPy 的隨機數生成器
    # 【連動檔案：server.py (初始全域權重), data.py (資料模組)】
    # 說明：伺服器若使用 NumPy 建立有種子的初始權重；以及資料前處理時，底層可能調用到 NumPy 的隨機打亂。
    np.random.seed(seed)

    # 3. 鎖定 PyTorch 神經網路的隨機性 (CPU 與 GPU)
    # 【連動檔案：pytorch_model.py (Dropout), client.py (小批次資料打亂與每次 fit 開始)】
    # 說明：控制模型初始化、訓練中的 Dropout 隨機關閉，以及 DataLoader 打亂資料 (shuffle) 的順序。
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 確保多張 GPU 都被鎖定

    # 4. 強制 PyTorch 關閉硬體加速的非確定性演算法
    # 【連動檔案：train_utils.py (GPU 運算核心)】
    # 說明：將 cuDNN 選項設為確定性模式，並關閉 benchmark，以防止 GPU 為了追求速度而犧牲小數點運算的一致性。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False