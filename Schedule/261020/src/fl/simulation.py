# src/fl/simulation.py 

import os  
import sys 
import numpy as np 
import torch 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.pytorch_model import DiabetesNet 
from src.fl.data import build_client_assets
from src.seed import set_global_seed
from src.train_utils import train, evaluate 

if __name__ == "__main__":
    
    # 執行種子
    EXECUTION_SEED = 42
    set_global_seed(EXECUTION_SEED)

    print("\n" + "="*60)  
    print("評估結果並更新本地模型 (三方比較)")  
    print("="*60)  
    
    partitions = ["0", "1", "2", "3", "4"]  
    train_loader, test_loader, val_loaders = {}, {}, {}
    
    # 預先載入所有分區的資料 (此處依舊保留完整 4 個回傳值)
    for cid in partitions:
        _, tr_loader, v_loader, te_loader = build_client_assets(cid)  
        train_loader[cid] = tr_loader  
        test_loader[cid] = te_loader  
        val_loaders[cid] = v_loader

    # 讀取訓練完成後的 global_model.npz (可以根據你的輸出路徑調整，若是由 run_experiment 產出，請指向該資料夾)
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../global_model.npz"))  
    loaded_data = np.load(MODEL_PATH)  
    final_ndarrays = [loaded_data[f] for f in loaded_data.files]

    print(f"{'Client':<8} | {'Model Type':<18} | {'Accuracy':<8} | {'F1 Score':<8} | {'ROC-AUC':<8}")
    print("-" * 65)  

    # 針對各分區執行三種模型版本的成效比較
    for cid in partitions:
        set_global_seed(EXECUTION_SEED + int(cid))
        input_dim = train_loader[cid].dataset.tensors[0].shape[1]  
        
        dummy_model = DiabetesNet(input_dim)  
        params_dict = zip(dummy_model.state_dict().keys(), final_ndarrays)  
        final_params = {k: torch.tensor(v) for k, v in params_dict}  

        # Local Only
        model_local = DiabetesNet(input_dim)  
        model_local = train(model_local, train_loader[cid], epochs=3, lr=0.005)  
        _, metrics_loc = evaluate(model_local, test_loader[cid])  
        
        # Global Only
        model_global = DiabetesNet(input_dim)  
        model_global.load_state_dict(final_params)  
        _, metrics_glob = evaluate(model_global, test_loader[cid])  
        
        # Personalized FL - 載入全域權重後，用local資料再微調
        model = DiabetesNet(input_dim)  
        model.load_state_dict(final_params)  
        model = train(model, train_loader[cid], epochs=3, lr=0.005) 
        _, metrics_pers = evaluate(model, test_loader[cid])  
        
        # 輸出各模型的測試結果
        print(f"Group {int(cid)+1} | {'Local Only':<18} | {metrics_loc['accuracy']:.4f}   | {metrics_loc['f1']:.4f}   | {metrics_loc['auc']:.4f}")
        print(f"Group {int(cid)+1} | {'Global Only':<18} | {metrics_glob['accuracy']:.4f}   | {metrics_glob['f1']:.4f}   | {metrics_glob['auc']:.4f}")
        print(f"Group {int(cid)+1} | {'Personalized FL':<18} | {metrics_pers['accuracy']:.4f}   | {metrics_pers['f1']:.4f}   | {metrics_pers['auc']:.4f}")
        print("-" * 65)

        #python -m src.fl.simulation