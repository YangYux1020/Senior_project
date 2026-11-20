# python check_repro.py
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import torch

# ==========================================
# 準備區：定義執行指令與權重讀取工具
# ==========================================
def run_cmd(seed: int, output_dir: str):
    """
    以獨立的子行程 (Subprocess) 啟動 run_experiment.py。
    這非常關鍵，因為若在同一個 Python 視窗連跑三次，記憶體殘留會破壞隨機性與 Ray 框架。
    """
    cmd = [
        sys.executable, "run_experiment.py",
        "--seed", str(seed),
        "--local_epochs", "3",       # 為了快速測試，只跑 3 個本地週期
        "--lr", "0.005",             
        "--rounds", "3",             # 為了快速測試，只跑 3 個聯邦回合
        "--output_folder", output_dir
    ]
    print(f"正在執行實驗 [種子: {seed}] -> 輸出至: {output_dir}")
    
    # subprocess.run 會開一個全新的終端機背景執行這段指令
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"實驗執行失敗 (種子 {seed})")
        sys.exit(1)

def load_weights(npz_path: str):
    """
    載入 global_model.npz 權重壓縮檔。
    將神經網路裡面一層層的權重矩陣，全部抽出來變成一個平坦的 NumPy 清單，方便後續直接做數學相減。
    """
    data = np.load(npz_path)
    return [data[file] for file in data.files]

if __name__ == "__main__":
    
    # ==========================================
    # 階段零：環境偵測與執行確認
    # ==========================================
    print("=" * 60)
    # 偵測是否抓得到 GPU，若有則一併顯示顯卡型號
    if torch.cuda.is_available():
        device_name = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        device_name = "CPU"
        
    print(f" 系統偵測：本次實驗即將使用的運算裝置為【{device_name}】")
    print("=" * 60)
    
    # 讓程式暫停，等待使用者輸入 (y/n)
    user_choice = input("請問是否確認執行這 3 場稽核實驗？ (y/n): ").strip().lower()
    if user_choice not in ['y', 'yes']:
        print("\n 已取消執行。請調整設定後再重新啟動。")
        sys.exit(0)  # 安全退出程式

    print("\n 開始執行實驗流程...\n")
       
    # 定義三次對照實驗的專屬輸出資料夾
    dir_seed0_run1 = "results_repro/seed0_run1"
    dir_seed0_run2 = "results_repro/seed0_run2"
    dir_seed1_run1 = "results_repro/seed1_run1"

    # ==========================================
    # 階段一：自動執行 3 場 3 回合的實驗
    # ==========================================
    print("=" * 60)
    print("階段一：自動執行 3 場 3 回合的實驗")
    print("=" * 60)
    
    # 啟動三次實驗：故意讓種子 0 跑兩次 (測絕對一致性)，種子 1 跑一次 (測隨機性是否有效)
    run_cmd(seed=0, output_dir=dir_seed0_run1)
    run_cmd(seed=0, output_dir=dir_seed0_run2)
    run_cmd(seed=1, output_dir=dir_seed1_run1)

    # ==========================================
    # 階段二：載入最終全域權重並計算最大絕對差異
    # ==========================================
    print("\n" + "=" * 60)
    print("階段二：載入最終全域權重並計算最大絕對差異")
    print("=" * 60)

    # 載入剛剛那三場實驗產出的最終全域模型權重 (global_model.npz)
    weights_s0_r1 = load_weights(os.path.join(dir_seed0_run1, "global_model.npz"))
    weights_s0_r2 = load_weights(os.path.join(dir_seed0_run2, "global_model.npz"))
    weights_s1_r1 = load_weights(os.path.join(dir_seed1_run1, "global_model.npz"))

    # 計算「相同種子 (種子 0 的兩次執行)」之間的權重差異
    # np.abs(w1 - w2) 會把兩套神經網路幾十萬個參數逐一相減並取絕對值，我們只看「最大」的那個誤差
    diff_same_seed = max(
        np.max(np.abs(w1 - w2)) for w1, w2 in zip(weights_s0_r1, weights_s0_r2)
    )

    # 計算「不同種子 (種子 0 與種子 1)」之間的權重差異
    diff_diff_seed = max(
        np.max(np.abs(w1 - w2)) for w1, w2 in zip(weights_s0_r1, weights_s1_r1)
    )

    print(f"相同種子 (Seed 0 vs Seed 0) 最終權重最大絕對差異: {diff_same_seed:.2e}")
    print(f"不同種子 (Seed 0 vs Seed 1) 最終權重最大絕對差異: {diff_diff_seed:.2e}")

    # ==========================================
    # 階段三：逐一比較相同種子之間的歷史檔數值
    # ==========================================
    print("\n" + "=" * 60)
    print("階段三：逐一比較相同種子之間的歷史檔數值")
    print("=" * 60)

    # 載入兩次種子 0 的全局歷史紀錄檔 (包含每輪的 Loss, Accuracy, F1, AUC)[cite: 13]
    hist_s0_r1 = pd.read_csv(os.path.join(dir_seed0_run1, "global_history.csv"))
    hist_s0_r2 = pd.read_csv(os.path.join(dir_seed0_run2, "global_history.csv"))

    print("比較兩次 Seed 0 的每輪指標數值：")
    print(f"{'Round':<8} | {'Metric':<12} | {'Run 1':<12} | {'Run 2':<12} | {'Diff':<12}")
    print("-" * 65)

    history_matched = True
    # 逐輪 (Round)、逐個指標 (Metric) 進行比對[cite: 13]
    for idx, row in hist_s0_r1.iterrows():
        r = row["round"]
        row2 = hist_s0_r2.iloc[idx]
        for col in ["val_loss", "accuracy", "f1", "auc"]:
            val1 = row[col]
            val2 = row2[col]
            diff = abs(val1 - val2)
            # 嚴格標準：只要小數點第 4 位以後有任何出入，就判定歷史紀錄不匹配[cite: 13]
            if diff > 1e-4:  
                history_matched = False
            print(f"{int(r):<8} | {col:<12} | {val1:<12.6f} | {val2:<12.6f} | {diff:<12.2e}")

    # ==========================================
    # 稽核結果總結
    # ==========================================
    print("\n" + "=" * 60)
    print("稽核結果總結")
    print("=" * 60)

    # 根據指導教授的稽核標準進行自動判定[cite: 13]
    if diff_same_seed < 1e-6 and history_matched:
        print("[可重現] 相同種子差異小於 1e-6，歷史紀錄小數點第 4 位完全相同，符合標準[cite: 13]。")
    elif diff_same_seed < 1e-3:
        print("[警告] 相同種子差異介於 1e-6 與 1e-3 之間，請檢查[cite: 13]。")
    else:
        print("[無法重現] 相同種子差異大於 1e-3，請依循疑難排解表檢查亂數設定[cite: 13]。")

    # 確保不同種子確實會造成不一樣的訓練軌跡[cite: 13]
    if diff_diff_seed > 1e-3:
        print("[差異顯著] 不同種子展現出不同的訓練軌跡，證明種子確實有發揮作用[cite: 13]。")
    else:
        print("[異常] 不同種子的差異過小，種子可能沒有成功傳入程式碼中[cite: 13]。")