import os
import subprocess
import argparse
import sys


def run_sweep(stage_name, epochs_list, lr_list, seeds):
    """
    自動化派發實驗任務 (FedAvg 超參數掃描與最終確認)
    """
    for E in epochs_list:
        for lr in lr_list:
            for seed in seeds:
                output_dir = f"results/{stage_name}/E{E}_lr{lr}/seed{seed}"

                success_flag = os.path.join(output_dir, "global_history.csv")
                if os.path.exists(success_flag):
                    print(f"略過已完成實驗: {output_dir}")
                    continue

                print(f"啟動實驗: [E={E}, lr={lr}, seed={seed}] -> {output_dir}")

                cmd = [
                    sys.executable, "run_experiment.py",
                    "--seed", str(seed),
                    "--local_epochs", str(E),
                    "--lr", str(lr),
                    "--rounds", "20",
                    "--output_folder", output_dir,
                ]

                result = subprocess.run(cmd)

                if result.returncode != 0:
                    print(f"實驗失敗: [E={E}, lr={lr}, seed={seed}]")
                    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["stageA", "stageB", "final"])
    args = parser.parse_args()

    seeds_to_run = [0, 1, 2, 3, 4]

    if args.stage == "stageA":
        print("=== 開始執行階段 A: 掃描 local_epochs ===")
        run_sweep("stageA", epochs_list=[1, 2, 3, 5, 10], lr_list=[0.01], seeds=seeds_to_run)

    elif args.stage == "stageB":
        best_E = 2
        print(f"=== 開始執行階段 B: 掃描 lr (固定 E*={best_E}) ===")
        run_sweep("stageB", epochs_list=[best_E], lr_list=[0.001, 0.003, 0.01, 0.03, 0.1], seeds=seeds_to_run)

    elif args.stage == "final":
        best_E = 2
        best_lr = 0.01
        final_seeds = [10, 11, 12, 13, 14]
        print(f"=== 開始執行最終確認階段: E*={best_E}, lr*={best_lr}, 種子 10-14 ===")
        run_sweep("final", epochs_list=[best_E], lr_list=[best_lr], seeds=final_seeds)
