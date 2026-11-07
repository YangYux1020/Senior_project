import os
import subprocess
import argparse
import sys

def run_sweep(stage_name, epochs_list, lr_list, seeds):
    """
    自動化派發實驗任務
    """
    for E in epochs_list:
        for lr in lr_list:
            for seed in seeds:
                # 定義該次實驗的專屬輸出資料夾
                output_dir = f"results/{stage_name}/E{E}_lr{lr}/seed{seed}"
                
                # 【中斷續跑防呆機制】檢查是否已經跑過了
                success_flag = os.path.join(output_dir, "global_history.csv")
                if os.path.exists(success_flag):
                    print(f"略過已完成實驗: {output_dir}")
                    continue
                
                print(f"啟動實驗: [E={E}, lr={lr}, seed={seed}] -> {output_dir}")
                
                # 準備執行指令
                cmd = [
                    sys.executable, "run_experiment.py",
                    "--seed", str(seed),
                    "--local_epochs", str(E),
                    "--lr", str(lr),
                    "--rounds", "20",  # 依指南設定 20 回合
                    "--output_folder", output_dir
                ]
                
                # 以獨立進程執行
                result = subprocess.run(cmd)
                
                if result.returncode != 0:
                    print(f"實驗失敗: [E={E}, lr={lr}, seed={seed}]")
                    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["stageA", "stageB", "final"])
    args = parser.parse_args()

    # 共通設定：使用 5 個固定的種子進行對照
    seeds_to_run = [0, 1, 2, 3, 4]

    if args.stage == "stageA":
        # 階段 A：掃描 local_epochs，固定 lr=0.01
        print("=== 開始執行階段 A: 掃描 local_epochs ===")
        run_sweep("stageA", epochs_list=[1, 2, 3, 5, 10], lr_list=[0.01], seeds=seeds_to_run)
        
    elif args.stage == "stageB":
        # 階段 B：掃描 lr，固定 local_epochs (請填入階段 A 選出的最佳值 E*)
        best_E = 2 # 經過 stageA 後，選擇 2 當作 E*
        print(f"=== 開始執行階段 B: 掃描 lr (固定 E*={best_E}) ===")
        run_sweep("stageB", epochs_list=[best_E], lr_list=[0.001, 0.003, 0.01, 0.03, 0.1], seeds=seeds_to_run)
        
    elif args.stage == "final":
        best_E = 2
        best_lr = 0.01
        final_seeds = [10, 11, 12, 13, 14]
        print(f"=== 開始執行最終確認階段: E*={best_E}, lr*={best_lr}, 種子 10-14 ===")
        run_sweep("final", epochs_list=[best_E], lr_list=[best_lr], seeds=final_seeds)   