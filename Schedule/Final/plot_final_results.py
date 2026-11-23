import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_final_comparisons():
    csv_path = "results/final/final_test_results.csv"
    output_dir = "results/final"
    
    if not os.path.exists(csv_path):
        print(f"找不到檔案: {csv_path}，請確認 final_compare.py 是否已順利執行並產出資料。")
        return

    # 讀取最終測試集的原始明細資料
    df = pd.read_csv(csv_path)

    # 設定繪圖風格
    sns.set_theme(style="whitegrid")
    # 設定調色盤 (Local-only, Global-only, Personalized)
    palette = {"Local-only": "#FF9999", "Global-only": "#66B2FF", "Personalized": "#99FF99"}

    print("開始繪製視覺化圖表...")

    # ==========================================
    # 圖表 1：F1-Score 比較 (著重於處理不平衡資料的能力)
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # 使用 seaborn 的 barplot，它會自動計算跨 seed 的平均值，並畫出 errorbar (標準差)
    ax1 = sns.barplot(
        data=df, 
        x="client", 
        y="f1", 
        hue="model_type", 
        palette=palette,
        errorbar="sd",
        capsize=0.1
    )
    
    plt.title("F1-Score Comparison across NHANES Partitions (Test Set)", fontsize=16, fontweight='bold')
    plt.xlabel("Client Partition (Survey Years)", fontsize=12)
    plt.ylabel("F1-Score (Mean ± Std)", fontsize=12)
    plt.ylim(0.75, 0.90)  # 依據數據表現微調 Y 軸範圍，讓差異更明顯
    plt.legend(title="Model Architecture", loc="lower right")
    
    # 儲存圖表
    f1_plot_path = os.path.join(output_dir, "f1_score_comparison.png")
    plt.tight_layout()
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()
    print(f"✅ F1-Score 長條圖已儲存至: {f1_plot_path}")

    # ==========================================
    # 圖表 2：AUC 比較 (著重於整體排序與預測能力)
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    ax2 = sns.barplot(
        data=df, 
        x="client", 
        y="auc", 
        hue="model_type", 
        palette=palette,
        errorbar="sd",
        capsize=0.1
    )
    
    plt.title("ROC-AUC Comparison across NHANES Partitions (Test Set)", fontsize=16, fontweight='bold')
    plt.xlabel("Client Partition (Survey Years)", fontsize=12)
    plt.ylabel("ROC-AUC (Mean ± Std)", fontsize=12)
    plt.ylim(0.88, 0.95)  # 依據數據表現微調 Y 軸範圍
    plt.legend(title="Model Architecture", loc="lower right")
    
    # 儲存圖表
    auc_plot_path = os.path.join(output_dir, "auc_comparison.png")
    plt.tight_layout()
    plt.savefig(auc_plot_path, dpi=300)
    plt.close()
    print(f"✅ AUC 長條圖已儲存至: {auc_plot_path}")
    
    print("\n視覺化步驟完成！您可以將這兩張圖直接放入畢業專題簡報中。")

if __name__ == "__main__":
    plot_final_comparisons()