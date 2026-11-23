"""自動化更新專題專案腳本 (update_project.py)

當 data/raw/balance_data_Group1~5.csv 資料集更新後，依序執行完整的更新流程：
  Step 0: 備份並清理舊的 results/ 與 results_repro/（避免 sweep.py 因偵測到舊檔而略過訓練）
  Step 1: 重新執行資料前處理，產出 data/processed/partition_Group1~5_{train,test}.parquet
  Step 2: 執行 100% 零誤差可重現性驗證 (check_repro.py)
  Step 3: 執行兩階段超參數調優掃描與彙整繪圖 (stageA, stageB + summarize.py)
  Step 4: 執行最終全域模型訓練 (sweep.py --stage final，並同步複製 seed10 權重至 ./global_model.npz)
  Step 5: 執行三方盲測、繪製最終對比圖並產出個人化微調消融結果
          (final_compare.py, plot_final_results.py, personalization_tuning_results.csv)
  Step 6 (選用 --include-advanced): 重新跑完少樣本稀缺進階實驗，
          產出 scarcity_results.csv

使用範例：
  python update_project.py                      # 執行核心更新流程 (Steps 0 ~ 5)
  python update_project.py --include-advanced   # 連同進階實驗 (Step 6) 一併更新
  python update_project.py --keep-existing      # 保留現有 results/ 續跑（不備份清空）
  python update_project.py --skip-repro --skip-sweep  # 略過重現性檢查與 Stage A/B 掃描
"""

import argparse
from datetime import datetime
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import preprocessing as pre
from src.fl.client import DiabetesClient
from src.fl.data import build_client_assets
from src.fl.server import get_strategy
from src.models.pytorch_model import DiabetesNet
from src.seed import set_global_seed
from src.train_utils import device, evaluate, train


YEAR_GROUPS = ["Group1", "Group2", "Group3", "Group4", "Group5"]
CLIENT_CIDS = ["0", "1", "2", "3", "4"]
FINAL_SEEDS = [10, 11, 12, 13, 14]
BEST_E = 2
BEST_LR = 0.01
NUM_ROUNDS = 20


def run_subprocess(cmd: list[str], stdin_text: str | None = None) -> None:
    """於專案根目錄執行子行程，若失敗則立即中止流程。"""
    print(f"\n[執行指令] {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        input=stdin_text,
        text=stdin_text is not None,
        check=True,
    )


def _seed_client_loader(train_loader, client_seed: int) -> None:
    """鎖定客戶端訓練種子與 DataLoader 洗牌順序。"""
    set_global_seed(client_seed)
    if hasattr(train_loader, "generator") and train_loader.generator is not None:
        train_loader.generator.manual_seed(client_seed)


def load_global_model_weights(model: torch.nn.Module, npz_path: str) -> torch.nn.Module:
    """載入保存在 .npz 檔案中的聯邦全域模型權重並寫入 PyTorch 模型。"""
    with np.load(npz_path) as data:
        weights = [data[f] for f in data.files]
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model


def _unique_backup_path(base_name: str, timestamp: str) -> str:
    candidate = os.path.join(PROJECT_ROOT, f"{base_name}_backup_{timestamp}")
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(PROJECT_ROOT, f"{base_name}_backup_{timestamp}_{suffix}")
        suffix += 1
    return candidate


def step0_backup_old_results(keep_existing: bool = False) -> None:
    """Step 0: 備份並清理舊的實驗輸出目錄，防止 sweep.py 略過已存在實驗。"""
    print("\n" + "=" * 75)
    print("Step 0: 檢查並備份舊有實驗結果")
    print("=" * 75)

    if keep_existing:
        print("已指定 --keep-existing：保留現有 results/ 與 results_repro/（將續跑未完成實驗）。")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for folder_name in ("results", "results_repro"):
        src_path = os.path.join(PROJECT_ROOT, folder_name)
        if os.path.exists(src_path):
            dst_path = _unique_backup_path(folder_name, timestamp)
            shutil.move(src_path, dst_path)
            print(f"已將舊目錄 {folder_name}/ 備份至 {os.path.basename(dst_path)}/")
        else:
            print(f"目錄 {folder_name}/ 不存在，無須備份。")


def step1_preprocess_data() -> None:
    """Step 1: 讀取 data/raw/balance_data_Group1~5.csv 並輸出標準化 .parquet 檔。"""
    print("\n" + "=" * 75)
    print("Step 1: 執行資料前處理 (以 Group2 [2003-2006 歷史基準] 訓練集 fit StandardScaler)")
    print("=" * 75)

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    raw_paths = {}
    for group_name in YEAR_GROUPS:
        csv_path = os.path.join(raw_dir, f"balance_data_{group_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到原始資料檔: {csv_path}")
        raw_paths[group_name] = csv_path

    x_train_base, x_test_base, base_scaler = pre.load_and_preprocess(
        raw_paths["Group2"], scaler=None, fit=True
    )
    x_train_base.to_parquet(os.path.join(processed_dir, "partition_Group2_train.parquet"))
    x_test_base.to_parquet(os.path.join(processed_dir, "partition_Group2_test.parquet"))
    print(f"已完成 Group2 前處理 -> train: {len(x_train_base)} 筆, test: {len(x_test_base)} 筆")

    for group_name in YEAR_GROUPS:
        if group_name == "Group2":
            continue
        x_train_scaled, x_test_scaled, _ = pre.load_and_preprocess(
            raw_paths[group_name], scaler=base_scaler, fit=False
        )
        x_train_scaled.to_parquet(os.path.join(processed_dir, f"partition_{group_name}_train.parquet"))
        x_test_scaled.to_parquet(os.path.join(processed_dir, f"partition_{group_name}_test.parquet"))
        print(f"已完成 {group_name} 前處理 -> train: {len(x_train_scaled)} 筆, test: {len(x_test_scaled)} 筆")


def step2_run_repro_check() -> None:
    """Step 2: 執行 100% 零誤差可重現性驗證 (check_repro.py)。"""
    print("\n" + "=" * 75)
    print("Step 2: 執行零誤差可重現性稽核 (check_repro.py)")
    print("=" * 75)
    run_subprocess([sys.executable, "check_repro.py"], stdin_text="y\n")


def step3_run_stage_sweeps() -> None:
    """Step 3: 執行 Stage A / Stage B 超參數掃描與彙整繪圖。"""
    print("\n" + "=" * 75)
    print("Step 3: 執行 Stage A 與 Stage B 超參數調優及彙整圖表生成")
    print("=" * 75)

    for stage in ("stageA", "stageB"):
        run_subprocess([sys.executable, "sweep.py", "--stage", stage])
        run_subprocess([sys.executable, "summarize.py", "--stage", stage])


def step4_run_final_training() -> None:
    """Step 4: 執行最終聯邦全域模型訓練 (Seeds 10~14) 並同步根目錄 global_model.npz。"""
    print("\n" + "=" * 75)
    print("Step 4: 執行最終確認階段全域模型訓練 (Seeds 10~14)")
    print("=" * 75)

    run_subprocess([sys.executable, "sweep.py", "--stage", "final"])

    seed10_npz = os.path.join(
        PROJECT_ROOT, "results", "final", f"E{BEST_E}_lr{BEST_LR}", "seed10", "global_model.npz"
    )
    root_npz = os.path.join(PROJECT_ROOT, "global_model.npz")
    if os.path.exists(seed10_npz):
        shutil.copy2(seed10_npz, root_npz)
        print(f"已同步最新全域模型權重至根目錄: {root_npz}")


def _generate_personalization_tuning_csv() -> None:
    """產出個人化微調強度消融實驗明細 results/final/personalization_tuning_results.csv。"""
    print("\n--- 產出個人化微調強度消融明細 (personalization_tuning_results.csv) ---")
    pers_settings = [
        ("Global-only (0 ep)", 0, 0.0),
        ("Gentle (E=1, lr=0.0005)", 1, 0.0005),
        ("Mild (E=1, lr=0.001)", 1, 0.001),
        ("Moderate (E=2, lr=0.001)", 2, 0.001),
        ("Original (E=3, lr=0.005)", 3, 0.005),
    ]

    output_dir = os.path.join(PROJECT_ROOT, "results", "final")
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for setting_name, pers_epochs, pers_lr in pers_settings:
        for seed in FINAL_SEEDS:
            npz_path = os.path.join(
                output_dir, f"E{BEST_E}_lr{BEST_LR}", f"seed{seed}", "global_model.npz"
            )
            if not os.path.exists(npz_path):
                raise FileNotFoundError(
                    f"找不到全域模型權重檔案: {npz_path}，請確認 sweep.py --stage final 已執行完成。"
                )

            for idx, cid in enumerate(CLIENT_CIDS):
                group_name = f"Group{int(cid) + 1}"
                client_seed = seed * 100 + idx
                set_global_seed(client_seed)

                _, train_loader, val_loader, test_loader = build_client_assets(cid)
                input_dim = train_loader.dataset.tensors[0].shape[1]

                model = DiabetesNet(input_dim=input_dim).to(device)
                model = load_global_model_weights(model, npz_path)

                if pers_epochs > 0:
                    _seed_client_loader(train_loader, client_seed)
                    model = train(model, train_loader, epochs=pers_epochs, lr=pers_lr)

                _, val_metrics = evaluate(model, val_loader)
                test_loss, test_metrics = evaluate(model, test_loader)

                records.append({
                    "setting": setting_name,
                    "pers_epochs": pers_epochs,
                    "pers_lr": pers_lr,
                    "seed": seed,
                    "client": group_name,
                    "val_auc": val_metrics["auc"],
                    "val_f1": val_metrics["f1"],
                    "val_acc": val_metrics["accuracy"],
                    "test_auc": test_metrics["auc"],
                    "test_f1": test_metrics["f1"],
                    "test_acc": test_metrics["accuracy"],
                    "test_loss": test_loss,
                })

    out_csv = os.path.join(output_dir, "personalization_tuning_results.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"已儲存個人化微調對照表: {out_csv}")


def step5_run_final_comparison_and_plots() -> None:
    """Step 5: 執行最終三方盲測、繪製比較圖並更新個人化微調消融結果。"""
    print("\n" + "=" * 75)
    print("Step 5: 執行最終三方盲測、長條圖繪製與個人化微調消融分析")
    print("=" * 75)

    run_subprocess([
        sys.executable, "final_compare.py",
        "--pers_epochs", "2",
        "--pers_lr", "0.001",
    ])
    run_subprocess([sys.executable, "plot_final_results.py"])
    _generate_personalization_tuning_csv()


def _run_scarcity_fl_simulation_worker(
    seed: int,
    sample_ratio: float,
    asymmetric_clinics: bool,
    output_folder: str,
) -> None:
    """於獨立子行程執行少樣本 / 不對稱情境之 Flower 聯邦學習模擬。"""
    import flwr as fl

    os.makedirs(output_folder, exist_ok=True)
    set_global_seed(seed)

    _, tr_loader, _, _ = build_client_assets("0")
    input_dim = tr_loader.dataset.tensors[0].shape[1]
    model_path = os.path.join(output_folder, "global_model.npz")

    def client_fn(cid: str):
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        idx = int(cid)
        eff_ratio = 0.05 if (asymmetric_clinics and idx in [0, 1, 2]) else sample_ratio
        client_seed = seed * 100 + idx
        model, train_loader, val_loader, _ = build_client_assets(
            cid, sample_ratio=eff_ratio, subsample_seed=client_seed
        )
        return DiabetesClient(cid, model, train_loader, val_loader).to_client()

    strategy = get_strategy(
        seed=seed,
        input_dim=input_dim,
        local_epochs=BEST_E,
        lr=BEST_LR,
        model_path=model_path,
    )
    client_resources = {
        "num_cpus": 2,
        "num_gpus": 0.2 if torch.cuda.is_available() else 0.0,
    }
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"num_cpus": 6},
    )

    pd.DataFrame(strategy.client_metrics_history).to_csv(
        os.path.join(output_folder, "client_metrics.csv"), index=False
    )

    global_history_list = []
    if history.metrics_distributed:
        losses = history.losses_distributed
        accuracies = history.metrics_distributed.get("accuracy", [])
        f1_scores = history.metrics_distributed.get("f1", [])
        aucs = history.metrics_distributed.get("auc", [])
        for i, (round_num, val_loss) in enumerate(losses):
            global_history_list.append({
                "round": round_num,
                "val_loss": val_loss,
                "accuracy": accuracies[i][1] if i < len(accuracies) else None,
                "f1": f1_scores[i][1] if i < len(f1_scores) else None,
                "auc": aucs[i][1] if i < len(aucs) else None,
            })
    pd.DataFrame(global_history_list).to_csv(
        os.path.join(output_folder, "global_history.csv"), index=False
    )


def _run_scarcity_experiments() -> None:
    """執行少樣本與偏鄉小診所不對稱醫療網實驗，產出 results/final/scarcity_results.csv。"""
    print("\n--- 進階實驗: 少樣本稀缺與偏鄉小診所賦能實驗 (scarcity_results.csv) ---")
    scenarios = [
        ("Asymmetric (小診所5% + 醫學中心100%)", "asymmetric", 1.0, True),
        ("5% 極稀缺 (每院 33~84 筆)", "ratio_0.05", 0.05, False),
        ("10% 稀缺 (每院 66~168 筆)", "ratio_0.10", 0.10, False),
        ("20% 低量 (每院 132~336 筆)", "ratio_0.20", 0.20, False),
        ("50% 中量 (每院 330~841 筆)", "ratio_0.50", 0.50, False),
        ("100% 完整資料 (每院 660~1682 筆)", "ratio_1.00", 1.00, False),
    ]
    total_local_epochs = NUM_ROUNDS * BEST_E
    pers_epochs = 2
    pers_lr = 0.001

    records = []
    for scenario_name, tag, ratio, asymmetric in scenarios:
        for seed in FINAL_SEEDS:
            if ratio == 1.0 and not asymmetric:
                npz_path = os.path.join(
                    PROJECT_ROOT, "results", "final", f"E{BEST_E}_lr{BEST_LR}", f"seed{seed}", "global_model.npz"
                )
                if not os.path.exists(npz_path):
                    raise FileNotFoundError(
                        f"找不到完整資料全域模型權重: {npz_path}，請先確認 Step 4 已執行。"
                    )
            else:
                sim_dir = os.path.join(PROJECT_ROOT, "results", "scarcity", tag, f"seed{seed}")
                npz_path = os.path.join(sim_dir, "global_model.npz")
                hist_path = os.path.join(sim_dir, "global_history.csv")
                if not (os.path.exists(npz_path) and os.path.exists(hist_path)):
                    cmd = [
                        sys.executable, __file__,
                        "--_internal-scarcity-sim",
                        "--seed", str(seed),
                        "--sample_ratio", str(ratio),
                        "--output_folder", sim_dir,
                    ]
                    if asymmetric:
                        cmd.append("--asymmetric_clinics")
                    run_subprocess(cmd)

            for idx, cid in enumerate(CLIENT_CIDS):
                group_name = f"Group{int(cid) + 1}"
                client_seed = seed * 100 + idx
                set_global_seed(client_seed)

                eff_ratio = 0.05 if (asymmetric and idx in [0, 1, 2]) else ratio
                _, train_loader, _, test_loader = build_client_assets(
                    cid, sample_ratio=eff_ratio, subsample_seed=client_seed
                )
                n_train = len(train_loader.dataset)
                input_dim = train_loader.dataset.tensors[0].shape[1]

                # 1. Local-only / 2. Global-only / 3. Personalized
                variants = [
                    ("Local-only", False, total_local_epochs, BEST_LR),
                    ("Global-only", True, 0, 0.0),
                    ("Personalized", True, pers_epochs, pers_lr),
                ]
                for model_type, load_global, epochs_val, lr_val in variants:
                    if epochs_val > 0:
                        _seed_client_loader(train_loader, client_seed)
                    model = DiabetesNet(input_dim=input_dim).to(device)
                    if load_global:
                        model = load_global_model_weights(model, npz_path)
                    if epochs_val > 0:
                        model = train(model, train_loader, epochs=epochs_val, lr=lr_val)
                    loss_val, metrics = evaluate(model, test_loader)
                    records.append({
                        "scenario": scenario_name,
                        "seed": seed,
                        "client": group_name,
                        "n_train": n_train,
                        "model_type": model_type,
                        "loss": loss_val,
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "auc": metrics["auc"],
                    })

    out_csv = os.path.join(PROJECT_ROOT, "results", "final", "scarcity_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"已儲存少樣本稀缺實驗明細表: {out_csv}")


def step6_run_advanced_experiments() -> None:
    """Step 6: 執行進階實驗 (少樣本稀缺賦能情境)。"""
    print("\n" + "=" * 75)
    print("Step 6: 執行進階實驗 (少樣本稀缺賦能)")
    print("=" * 75)
    _run_scarcity_experiments()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="更動 data/raw/balance_data_Group1~5.csv 後，自動化執行專題專案全流程更新"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="保留現有的 results/ 與 results_repro/（續跑未完成的實驗，不進行備份清空）",
    )
    parser.add_argument(
        "--skip-repro",
        action="store_true",
        help="略過 Step 2 的可重現性檢查 (check_repro.py)",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="略過 Step 3 的 Stage A / Stage B 超參數掃描與繪圖",
    )
    parser.add_argument(
        "--include-advanced",
        action="store_true",
        help="額外執行 Step 6 進階實驗以重新產生 scarcity_results.csv",
    )

    # 內部子行程專用旗標（確保 Flower + Ray 模擬於獨立行程中執行以維持 100% 可重現性）
    parser.add_argument("--_internal-scarcity-sim", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--sample_ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--asymmetric_clinics", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output_folder", type=str, default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args._internal_scarcity_sim:
        _run_scarcity_fl_simulation_worker(
            seed=args.seed,
            sample_ratio=args.sample_ratio,
            asymmetric_clinics=args.asymmetric_clinics,
            output_folder=args.output_folder,
        )
        return

    print("===========================================================================")
    print("開始執行專題自動化更新流程 (update_project.py)")
    print(f"專案路徑: {PROJECT_ROOT}")
    print(f"運算裝置: {device}")
    print("===========================================================================")

    step0_backup_old_results(keep_existing=args.keep_existing)
    step1_preprocess_data()

    if not args.skip_repro:
        step2_run_repro_check()
    else:
        print("\n[略過] Step 2: 可重現性稽核 (--skip-repro)")

    if not args.skip_sweep:
        step3_run_stage_sweeps()
    else:
        print("\n[略過] Step 3: Stage A / Stage B 超參數掃描 (--skip-sweep)")

    step4_run_final_training()
    step5_run_final_comparison_and_plots()

    if args.include_advanced:
        step6_run_advanced_experiments()
    else:
        print("\n[提示] Step 6 進階實驗未執行；若需更新 scarcity_results.csv，請加上 --include-advanced。")

    print("\n" + "=" * 75)
    print("專題更新完成！請執行以下指令啟動或重新整理展示系統：")
    print("  streamlit run app.py")
    print("（若網頁已開啟，請於右上角選單點擊 Clear cache 或按鍵盤 C 清除快取）")
    print("=" * 75)


if __name__ == "__main__":
    main()
