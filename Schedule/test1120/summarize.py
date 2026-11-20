import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def summarize_stage(stage_name):
    stage_dir = f"results/{stage_name}"
    if not os.path.exists(stage_dir):
        print(f"找不到目錄 {stage_dir}")
        return

    # 儲存繪圖用資料：設定名稱 -> 平均 AUC 陣列
    plot_data_rounds = {}
    plot_data_epochs = {}

    # 1. 走訪該階段下所有的參數組合資料夾 (例如 E2_lr0.01)
    for setting_name in os.listdir(stage_dir):
        setting_dir = os.path.join(stage_dir, setting_name)
        if not os.path.isdir(setting_dir):
            continue

        all_seeds_auc = []
        E_val = int(setting_name.split('_')[0].replace('E', '')) # 萃取 E 值供繪圖用
        
        # 2. 讀取該組合下 5 個種子的歷史資料
        for seed in [0, 1, 2, 3, 4]:
            history_path = os.path.join(setting_dir, f"seed{seed}", "global_history.csv")
            if os.path.exists(history_path):
                df = pd.read_csv(history_path)
                all_seeds_auc.append(df['auc'].values)
            else:
                print(f"缺少種子資料: {history_path}")

        if len(all_seeds_auc) == 5:
            # 將 5 個種子的曲線疊加成矩陣 (5 x 20)
            auc_matrix = np.vstack(all_seeds_auc)
            
            # 計算各回合的平均值與標準差
            mean_auc = np.mean(auc_matrix, axis=0)
            std_auc = np.std(auc_matrix, axis=0)
            
            # 記錄最終表現 (最後一輪)
            final_mean = mean_auc[-1]
            final_std = std_auc[-1]
            print(f"【{setting_name}】最終平均 AUC: {final_mean:.4f} ± {final_std:.4f}")

            plot_data_rounds[setting_name] = (mean_auc, std_auc)
            plot_data_epochs[setting_name] = (mean_auc, std_auc, E_val)

    # 3. 繪製圖表 1：AUC vs 通訊回合數
    plt.figure(figsize=(10, 6))
    for setting_name, (mean_auc, std_auc) in plot_data_rounds.items():
        rounds = np.arange(1, len(mean_auc) + 1)
        plt.plot(rounds, mean_auc, label=setting_name)
        plt.fill_between(rounds, mean_auc - std_auc, mean_auc + std_auc, alpha=0.2)
    plt.title(f"{stage_name}: Validation AUC vs Communication Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Validation AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stage_dir, "auc_vs_rounds.png"))
    plt.close()

    # 4. 繪製圖表 2：AUC vs 總本地訓練週期數
    plt.figure(figsize=(10, 6))
    for setting_name, (mean_auc, std_auc, E_val) in plot_data_epochs.items():
        # X 軸為：當前輪數 * 本地週期數 E
        total_epochs = np.arange(1, len(mean_auc) + 1) * E_val
        plt.plot(total_epochs, mean_auc, label=setting_name, marker='o')
    plt.title(f"{stage_name}: Validation AUC vs Total Local Epochs")
    plt.xlabel("Total Local Epochs (Rounds * E)")
    plt.ylabel("Validation AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stage_dir, "auc_vs_total_epochs.png"))
    plt.close()
    
    print(f"\n彙整完成！圖表已儲存至 {stage_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["stageA", "stageB"])
    args = parser.parse_args()
    
    summarize_stage(args.stage)