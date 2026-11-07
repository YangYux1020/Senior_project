`==============================================================================================================10/7進度==============================================================================================================`

-----------------------
1. 各自的功能
    (1)preprocessing.py:
    
        資料前處理，把原始 NHANES資料，處理缺失值、正規化，
        並切分成 5 個parquet檔

    (2)pytorch_model.py:
    
        定義神經網路架構 DiabetesNet(MLP)，包括有幾層網路、每層的node數量跟Dropout的'比例。

    (3)train_utils.py:
    
        train 跟 evaluate，計算 Loss、反向傳播梯度更新，
        計算 Accuracy、F1、ROC-AUC 都在這裡算。

    (4)client.py:
    
        定義 DiabetesClient 模板，負責處理server.py傳進來的numpy格式權重，轉換成train_utils要用的tensor格式，
        從simulation.py 接收資料以及跟server.py 接收最新的權重指令，然後丟給 train_utils.py 訓練，最後再把更新後的權重(再轉換成numpy格式)回傳給server.py。

    (5)server.py:
    
        定義 Flower 的演算法(FedAvg or FedProx)，並決定訓練策略(local epoch、learning rate...)等等規則
        把client回傳的numpy權重進行FedAvg平均，最後把10倫模擬後的global model參數，儲存並打包成global_model.npz（SaveModelStrategy)
        --->後續才能在simulation.py讀出並計算Personalized FL。

    (6)simulation.py:
    
        負責全部的流程控制，啟動以及結束，主要有
            1.設定環境(Random Seed)
            
            2.依照需求建立DiabetesClient模板的client_fn
            
            3.讀取Parquet 檔，轉換為 Tensor，打包成 PyTorch DataLoader給client.py
            
            4.串聯 server.py 的策略，並用Ray來模擬框架，並開始跑 10 輪測試。

            5.成效對比，讀取剛存檔的global_model.npz，以及local的計算結果，丟給train_utils.py算出Personalized FL，
            最後對三個模型進行比較並輸出。

-----------------------

2. 之間的關係 

    資料準備：
    
        preprocessing.py 執行，產出 Parquet 檔案。資料存在硬碟。

    模擬啟動：
    
        執行simulation.py，先讀取 server.py 取得聚合規則，接著啟動Flower模擬。

    計算跟傳輸：
    
        伺服器發送初始權重。simulation.py 會動態生成 5 個 client_fn

        client.py 載入 pytorch_model.py 建立模型外殼。

        client.py 將global權重塞入模型，呼叫 train_utils.py 進行訓練。

        訓練完成後，client.py 把新的權重回傳給 server.py。

        server.py 聚合這 5 份權重，產生下一輪的全新全域模型。Run 10 次。   

    評估與輸出：
    
        10 輪結束後，server.py 將最終全域模型儲存。simulation.py 攔截這個最終模型，配合local結果，算出Personalized FL，
        最後比較然後輸出。

-----------------------


3. 預測分數
    (1)Accuracy:
        在所有病患中，總共猜中的比例，包含正確抓出有病，以及正確確認沒病，但容易產生誤導跟偏差。

    (2)F1-Score:
        兼顧Precision(誤診)、Recall(漏診)的評估指標

    (3)ROC-AUC:
        評估模型區分病患與健康人群的潛力，在一個有病一個沒病的樣本中，把有病的患病風險排在沒病的之上的機率。


`==============================================================================================================10/14進度==============================================================================================================`
`更新:`
(1)seed.py (新增):
設定環境(Random Seed)，強制把 Python、NumPy、PyTorch等等...亂數鎖死，確保每次實驗跑出來的結果一模一樣。

(2)data.py (新增):
    接手前處理的資料，用固定的 SPLIT_SEED 切分成各個 client 要用的 train_loader 跟 val_loader(驗證集)。

(3)client.py (更新):
    
    每次訓練前動態算出專屬的 seed 重設亂數，丟給 train_utils.py 訓練，評估時改用 val_loader，最後再把更新後的權重(加上 client ID)回傳給server.py。

(4)server.py (更新):
    定義 Flower 的演算法，並決定訓練策略(local epoch、learning rate...)等等規則。
    把client回傳的numpy權重「依照 client ID 排序」後進行 FedAvg 平均，把每輪的指標(Accuracy、F1、AUC)存成 CSV。
    最後把10倫模擬後的global model參數，儲存並打包成global_model.npz。

(5)run_experiment.py (新增):
    取代 simulation.py 成為啟動點。負責從終端機接收參數(seed、lr、epoch...)，啟動 Flower 模擬跑 10 輪測試。
    結束後輸出 config.json 跟 global_history.csv 紀錄檔。

(6)check_repro.py (新增):
    自動化測試工具，啟動 3 次 run_experiment.py (種子 0 跑兩次，種子 1 跑一次)，載入最後的 global_model.npz 比較權重差異，證明實驗真的可以重現( check_repro.py terminal 輸入 python check_repro.py )。

(7)simulation.py (在二部分會變成 final_compare.py/未完成):
    拔除 Flower 模擬啟動的功能，現在只負責最後的成效對比。讀取剛存檔的global_model.npz，以及local的計算


`更新流程:`

資料準備與環境鎖定：

    preprocessing.py 執行，產出 Parquet 檔案。資料存在硬碟。
    執行實驗前，先呼叫 seed.py 鎖定亂數環境。

模擬啟動：

    執行 run_experiment.py，透過 data.py 準備資料，讀取 server.py 取得聚合規則，接著啟動Flower模擬。

計算跟傳輸：

    伺服器發送初始權重。run_experiment.py 會動態生成 5 個 client_fn。
    client.py 載入 pytorch_model.py 建立模型外殼。
    client.py 將global權重塞入模型，動態重設 seed 後呼叫 train_utils.py 進行訓練。
    訓練完成後，client.py 把新的權重跟 client ID 回傳給 server.py。
    server.py 依照 ID 排序後聚合這 5 份權重，產生下一輪的全新全域模型。Run 10 次。

評估與輸出：
    10 輪結束後，server.py 將最終全域模型儲存成 npz，run_experiment.py 輸出 CSV 紀錄檔。
    接著執行 simulation.py 攔截這個最終模型，配合local結果，在 test_loader 上算出Personalized FL，最後比較然後輸出。
    另外可用 check_repro.py 跑三次實驗比對權重，證明系統可重現。


`==============================================================================================================10/20進度==============================================================================================================`
`更新:`