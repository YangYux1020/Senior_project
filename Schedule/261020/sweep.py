"""
接收階段名稱，並對組合與種子進行迴圈。
每個組合都以獨立程序啟動 run_experiment.py，並寫入以階段、local_epochs、lr 與種子命名的資料夾（例如 results/stageA/E2_lr0.01/seed3）。
若某次執行的歷史檔已存在就略過，使中斷的掃描可以直接重新啟動。
需要四個階段：stageA、stageB（以最佳 local_epochs 為參數）、stageC（接收 local_epochs 與 lr 的清單）以及 final（接收選定的 local_epochs 與 lr，並使用種子 10 至 14）。
"""
