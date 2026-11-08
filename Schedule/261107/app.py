import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from src import preprocessing as pre

# streamlit run app.py

# ==========================================
# 網頁基本設定
# ==========================================
st.set_page_config(
    page_title="聯邦學習慢性病風險預測系統",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 簡報級極簡視覺樣式
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1 { font-size: 2.0rem !important; font-weight: 800 !important; margin-bottom: 0.4rem !important; }
    h2 { font-size: 1.45rem !important; font-weight: 700 !important; margin-top: 1rem !important; }
    h3 { font-size: 1.15rem !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 12px 16px;
    }
    .tag-box {
        background: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURE_LABELS = {
    "RIDAGEYR": "年齡 (RIDAGEYR, 歲)",
    "RIAGENDR": "性別 (RIAGENDR)",
    "LBXGH": "糖化血色素 (LBXGH, %)",
    "LBXGLU": "空腹血糖 (LBXGLU, mg/dL)",
    "LBXHGB": "血紅素 (LBXHGB, g/dL)",
    "LBXRBCSI": "紅血球數 (LBXRBCSI, 10⁶/μL)",
    "LBXHCT": "血球容積比 (LBXHCT, %)",
    "LBDHDD": "高密度膽固醇 (LBDHDD, mg/dL)",
    "LBDLDL": "低密度膽固醇 (LBDLDL, mg/dL)",
    "URXUMA": "尿液微量白蛋白 (URXUMA, μg/mL)",
    "LBXSCR": "血清肌酸酐 (LBXSCR, mg/dL)",
    "LBXSASSI": "天門冬胺酸轉胺酶 AST (LBXSASSI, U/L)",
    "LBXSGTSI": "丙麩胺酸轉移酶 GGT (LBXSGTSI, U/L)",
    "GH_GLU_Interact": "長短期血糖乘積 (GH_GLU_Interact)",
    "UACR_Proxy": "白蛋白肌酸酐比 (UACR_Proxy)",
    "Lipid_Ratio": "壞好膽固醇比 (Lipid_Ratio)",
}

RAW_FEATURES = [
    "RIDAGEYR", "RIAGENDR", "LBXGH", "LBXHGB", "LBXGLU",
    "LBXRBCSI", "URXUMA", "LBDHDD", "LBDLDL", "LBXSASSI",
    "LBXSGTSI", "LBXSCR", "LBXHCT",
]
ALL_FEATURES = RAW_FEATURES + ["GH_GLU_Interact", "UACR_Proxy", "Lipid_Ratio"]


# ==========================================
# 資料與模型載入快取
# ==========================================
@st.cache_data
def load_raw_partitions():
    frames = []
    for i in range(1, 6):
        path = f"data/raw/balance_data_Group{i}.csv"
        df = pd.read_csv(path)
        df = df[df["DIQ010"] != 9.0].reset_index(drop=True)
        df["DIQ010"] = df["DIQ010"].replace({3.0: 1.0, 2.0: 0.0})
        df["GH_GLU_Interact"] = df["LBXGH"] * df["LBXGLU"]
        df["UACR_Proxy"] = df["URXUMA"] / df["LBXSCR"]
        df["Lipid_Ratio"] = df["LBDLDL"] / df["LBDHDD"]
        df["Group"] = f"Group {i}"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def get_base_scaler_and_medians():
    _, _, scaler = pre.load_and_preprocess(
        "data/raw/balance_data_Group1.csv", scaler=None, fit=True
    )
    raw_df = load_raw_partitions()
    healthy_medians = raw_df[raw_df["DIQ010"] == 0.0][RAW_FEATURES].median().to_dict()
    return scaler, healthy_medians


@st.cache_data
def compute_rf_and_corr():
    raw_df = load_raw_partitions()
    imp_records = []
    for g in ["Group 1", "Group 5"]:
        sub = raw_df[raw_df["Group"] == g]
        X = sub[ALL_FEATURES]
        y = sub["DIQ010"]
        rf = RandomForestClassifier(n_estimators=150, random_state=42)
        rf.fit(X, y)
        for feat, val in zip(ALL_FEATURES, rf.feature_importances_):
            imp_records.append({"Group": g, "Feature": feat, "Importance": val})
    imp_df = pd.DataFrame(imp_records)
    corr_g1 = raw_df[raw_df["Group"] == "Group 1"][ALL_FEATURES].corr()
    corr_g5 = raw_df[raw_df["Group"] == "Group 5"][ALL_FEATURES].corr()
    return imp_df, corr_g1, corr_g5


@st.cache_data
def load_stage_histories(stage_name: str):
    stage_dir = f"results/{stage_name}"
    curve_rows = []
    summary_rows = []
    if not os.path.exists(stage_dir):
        return pd.DataFrame(), pd.DataFrame()

    settings = sorted(
        [d for d in os.listdir(stage_dir) if os.path.isdir(os.path.join(stage_dir, d))],
        key=lambda s: (int(s.split("_")[0].replace("E", "")), float(s.split("_")[1].replace("lr", ""))),
    )

    for setting in settings:
        setting_dir = os.path.join(stage_dir, setting)
        e_val = int(setting.split("_")[0].replace("E", ""))
        lr_val = float(setting.split("_")[1].replace("lr", ""))
        dfs = []
        for seed in range(5):
            hp = os.path.join(setting_dir, f"seed{seed}", "global_history.csv")
            if os.path.exists(hp):
                d = pd.read_csv(hp)
                d["seed"] = seed
                dfs.append(d)
        if len(dfs) == 5:
            full = pd.concat(dfs, ignore_index=True)
            agg = full.groupby("round")[["auc", "f1", "accuracy", "val_loss"]].agg(["mean", "std"]).reset_index()
            for _, r in agg.iterrows():
                rd = int(r[("round", "")])
                curve_rows.append({
                    "setting": setting,
                    "E": e_val,
                    "lr": lr_val,
                    "round": rd,
                    "total_epochs": rd * e_val,
                    "auc_mean": r[("auc", "mean")],
                    "auc_std": r[("auc", "std")],
                    "f1_mean": r[("f1", "mean")],
                    "f1_std": r[("f1", "std")],
                    "acc_mean": r[("accuracy", "mean")],
                    "acc_std": r[("accuracy", "std")],
                    "loss_mean": r[("val_loss", "mean")],
                    "loss_std": r[("val_loss", "std")],
                })
            mean_curve = agg[("auc", "mean")].values
            gain = mean_curve[-1] - mean_curve[0]
            r95 = int(np.argmax(mean_curve >= mean_curve[0] + 0.95 * gain)) + 1 if gain > 0 else 1
            last = full[full["round"] == 20]
            summary_rows.append({
                "參數組合": setting,
                "E": e_val,
                "lr": lr_val,
                "總本地週期 (20×E)": 20 * e_val,
                "最終驗證 AUC": f"{last['auc'].mean():.4f} ± {last['auc'].std(ddof=0):.4f}",
                "最終驗證 F1": f"{last['f1'].mean():.4f} ± {last['f1'].std(ddof=0):.4f}",
                "最終驗證 Accuracy": f"{last['accuracy'].mean():.4f} ± {last['accuracy'].std(ddof=0):.4f}",
                "95% 收斂回合": f"Round {r95}",
                "_auc_val": last["auc"].mean(),
            })
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


@st.cache_data
def load_fedprox_curves():
    path = "results/final/fedprox_summary_curves.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_pers_tuning():
    path = "results/final/personalization_tuning_results.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def predict_with_weights(x_scaled: np.ndarray, d: dict) -> float:
    """使用權重字典進行 DiabetesNet 前向傳播推論"""
    x = x_scaled.astype(np.float32) @ d["arr_0"].T + d["arr_1"]
    x = (x - d["arr_4"]) / np.sqrt(d["arr_5"] + 1e-5) * d["arr_2"] + d["arr_3"]
    x = np.maximum(x, 0.0)
    x = x @ d["arr_7"].T + d["arr_8"]
    x = (x - d["arr_11"]) / np.sqrt(d["arr_12"] + 1e-5) * d["arr_9"] + d["arr_10"]
    x = np.maximum(x, 0.0)
    logit = float((x @ d["arr_14"].T + d["arr_15"]).reshape(-1)[0])
    return float(1.0 / (1.0 + np.exp(-logit)))


def predict_with_npz(x_scaled: np.ndarray, npz_path: str) -> float:
    """使用真實訓練完成之 DiabetesNet (.npz) 權重進行前向傳播推論"""
    d = np.load(npz_path)
    return predict_with_weights(x_scaled, d)


@st.cache_data
def get_personalized_weights_dict(pers_epochs: int = 2, pers_lr: float = 0.001):
    """載入全域模型並對各分區 (Group 1 ~ 5) 執行真實個人化微調 (Adam + BatchNorm + Dropout)"""
    from sklearn.model_selection import train_test_split
    _, _, base_scaler = pre.load_and_preprocess("data/raw/balance_data_Group1.csv", scaler=None, fit=True)
    train_sets = {}
    for i in range(1, 6):
        tr_full, _, _ = pre.load_and_preprocess(f"data/raw/balance_data_Group{i}.csv", scaler=base_scaler, fit=False)
        tr_sub, _ = train_test_split(tr_full, test_size=0.15, random_state=42, stratify=tr_full["DIQ010"])
        train_sets[f"Group {i}"] = (
            tr_sub.drop("DIQ010", axis=1).values.astype(np.float32),
            tr_sub["DIQ010"].values.astype(np.float32).reshape(-1, 1),
        )

    pers_models = {}
    for g_idx, g_name in enumerate(["Group 1", "Group 2", "Group 3", "Group 4", "Group 5"]):
        X_tr, y_tr = train_sets[g_name]
        pers_models[g_name] = {}
        for s in [10, 11, 12, 13, 14]:
            npz_p = f"results/final/E2_lr0.01/seed{s}/global_model.npz"
            if not os.path.exists(npz_p):
                continue
            raw_npz = np.load(npz_p)
            d = {k: raw_npz[k].astype(np.float32).copy() for k in raw_npz.files}
            rng = np.random.RandomState(s * 100 + g_idx)
            param_keys = ["arr_0", "arr_1", "arr_2", "arr_3", "arr_7", "arr_8", "arr_9", "arr_10", "arr_14", "arr_15"]
            m_adam = {k: np.zeros_like(d[k]) for k in param_keys}
            v_adam = {k: np.zeros_like(d[k]) for k in param_keys}
            t_step = 0
            n = len(X_tr)
            for _ in range(pers_epochs):
                perm = rng.permutation(n)
                for start in range(0, n, 32):
                    idx = perm[start:start + 32]
                    xb, yb = X_tr[idx], y_tr[idx]
                    B = len(xb)
                    if B <= 1:
                        continue
                    t_step += 1
                    z1 = xb @ d["arr_0"].T + d["arr_1"]
                    m1, v1 = z1.mean(axis=0), z1.var(axis=0)
                    d["arr_4"] = 0.9 * d["arr_4"] + 0.1 * m1
                    d["arr_5"] = 0.9 * d["arr_5"] + 0.1 * (v1 * B / (B - 1))
                    std1 = np.sqrt(v1 + 1e-5)
                    z1_hat = (z1 - m1) / std1
                    bn1 = d["arr_2"] * z1_hat + d["arr_3"]
                    a1 = np.maximum(bn1, 0.0)
                    mask1 = (rng.rand(*a1.shape) >= 0.2).astype(np.float32) / 0.8
                    d1_out = a1 * mask1

                    z2 = d1_out @ d["arr_7"].T + d["arr_8"]
                    m2, v2 = z2.mean(axis=0), z2.var(axis=0)
                    d["arr_11"] = 0.9 * d["arr_11"] + 0.1 * m2
                    d["arr_12"] = 0.9 * d["arr_12"] + 0.1 * (v2 * B / (B - 1))
                    std2 = np.sqrt(v2 + 1e-5)
                    z2_hat = (z2 - m2) / std2
                    bn2 = d["arr_9"] * z2_hat + d["arr_10"]
                    a2 = np.maximum(bn2, 0.0)
                    mask2 = (rng.rand(*a2.shape) >= 0.2).astype(np.float32) / 0.8
                    d2_out = a2 * mask2

                    logits = d2_out @ d["arr_14"].T + d["arr_15"]
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                    dlogits = (probs - yb) / B

                    grads = {}
                    grads["arr_14"] = dlogits.T @ d2_out
                    grads["arr_15"] = dlogits.sum(axis=0)
                    dd2 = dlogits @ d["arr_14"]
                    dbn2 = (dd2 * mask2) * (bn2 > 0)
                    grads["arr_9"] = (dbn2 * z2_hat).sum(axis=0)
                    grads["arr_10"] = dbn2.sum(axis=0)
                    dz2_hat = dbn2 * d["arr_9"]
                    dz2 = (1.0 / B) / std2 * (B * dz2_hat - dz2_hat.sum(axis=0) - z2_hat * (dz2_hat * z2_hat).sum(axis=0))

                    grads["arr_7"] = dz2.T @ d1_out
                    grads["arr_8"] = dz2.sum(axis=0)
                    dd1 = dz2 @ d["arr_7"]
                    dbn1 = (dd1 * mask1) * (bn1 > 0)
                    grads["arr_2"] = (dbn1 * z1_hat).sum(axis=0)
                    grads["arr_3"] = dbn1.sum(axis=0)
                    dz1_hat = dbn1 * d["arr_2"]
                    dz1 = (1.0 / B) / std1 * (B * dz1_hat - dz1_hat.sum(axis=0) - z1_hat * (dz1_hat * z1_hat).sum(axis=0))
                    grads["arr_0"] = dz1.T @ xb
                    grads["arr_1"] = dz1.sum(axis=0)

                    for k in param_keys:
                        m_adam[k] = 0.9 * m_adam[k] + 0.1 * grads[k]
                        v_adam[k] = 0.999 * v_adam[k] + 0.001 * (grads[k] ** 2)
                        m_hat = m_adam[k] / (1.0 - 0.9 ** t_step)
                        v_hat = v_adam[k] / (1.0 - 0.999 ** t_step)
                        d[k] -= pers_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            pers_models[g_name][s] = d
    return pers_models


def build_feature_vector(raw_dict: dict, scaler) -> np.ndarray:
    row = dict(raw_dict)
    row["GH_GLU_Interact"] = row["LBXGH"] * row["LBXGLU"]
    row["UACR_Proxy"] = row["URXUMA"] / max(row["LBXSCR"], 0.01)
    row["Lipid_Ratio"] = row["LBDLDL"] / max(row["LBDHDD"], 1.0)
    df_one = pd.DataFrame([[row[col] for col in ALL_FEATURES]], columns=ALL_FEATURES)
    return scaler.transform(df_one)


# ==========================================
# 側邊欄導覽
# ==========================================
st.sidebar.title("聯邦式學習應用於慢性病風險預測")
st.sidebar.caption("Federated Learning on Multi-Cycle NHANES")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "展示章節",
    [
        "1. 研究動機與 Non-IID 數據",
        "2. 聯邦學習架構與可重現性",
        "3. 兩階段超參數調優",
        "4. 最終盲測效能對決",
        "5. Live Demo 臨床即時推論",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **系統規格速覽**
    - **資料集**：NHANES (5 個時間分區)
    - **特徵維度**：16 維 (13 原始 + 3 衍生)
    - **資料切分**：68% 訓練 / 12% 驗證 / 20% 測試
    - **模型架構**：`DiabetesNet` (MLP + BN + Dropout)
    - **最佳參數**：$E^* = 2,\\; \\text{lr}^* = 0.01$
    - **聯邦聚合**：`FedAvg` / `FedProx` ($\\mu \\in [0, 0.1]$)
    - **個人化微調**：$E_p = 2,\\; \\text{lr}_p = 0.001$
    """
)
st.sidebar.markdown("---")
st.sidebar.info("輔仁大學資訊工程學系\n畢業專題（楊宇翔、鄭智嶸、林恒箴）")


# ==========================================
# 章節 1：研究動機與 Non-IID 數據
# ==========================================
if page == "1. 研究動機與 Non-IID 數據":
    st.title("1. 研究動機：醫療隱私限制與跨世代資料分佈")

    raw_df = load_raw_partitions()

    # 頂部關鍵指標卡
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("醫療時間分區", "5 Groups", "2003 ~ 2023 年")
    k2.metric("總樣本規模", f"{len(raw_df):,} 筆", "Group 1 (972) ~ Group 5 (2,474)")
    k3.metric("輸入特徵維度", "16 項", "13 生理指標 + 3 交互特徵")
    k4.metric("各分區健康比例", "1 : 1", "50.0% 風險 / 50.0% 健康")

    st.markdown("---")

    # 三大核心設計標題卡
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="tag-box">① 隱私法規與跨院資料限制 (PDPA / GDPR)</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="tag-box">② 二元風險定義與 1 : 1 類別平衡設計</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="tag-box">③ 防資料外洩之統一標準化 (Group 1 基準)</div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📊 跨分區樣本量與類別平衡",
        "📈 跨分區生理特徵分佈 (Feature Drift)",
        "🔥 跨時期特徵重要度與相關性位移",
    ])

    with tab1:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            grp_counts = (
                raw_df.groupby(["Group", "DIQ010"])
                .size()
                .reset_index(name="Count")
            )
            grp_counts["Label"] = grp_counts["DIQ010"].map({0.0: "健康 (0.0)", 1.0: "高風險 (1.0)"})
            fig_cnt = px.bar(
                grp_counts,
                x="Group",
                y="Count",
                color="Label",
                barmode="group",
                text="Count",
                color_discrete_map={"健康 (0.0)": "#3b82f6", "高風險 (1.0)": "#ef4444"},
                title="各醫療分區 (Group 1 ~ 5) 正負樣本數分佈 (樣本量高度不均 + 類別 1:1 平衡)",
            )
            fig_cnt.update_layout(height=400, xaxis_title="", yaxis_title="樣本數", legend_title="類別")
            st.plotly_chart(fig_cnt, use_container_width=True)

        with col_b:
            split_rows = []
            for i in range(1, 6):
                n_total = len(raw_df[raw_df["Group"] == f"Group {i}"])
                n_test = int(np.ceil(n_total * 0.2))
                n_tr_full = n_total - n_test
                n_val = int(np.ceil(n_tr_full * 0.15))
                n_train = n_tr_full - n_val
                split_rows.append({
                    "分區": f"Group {i}",
                    "總樣本": n_total,
                    "訓練集 (68%)": n_train,
                    "驗證集 (12%)": n_val,
                    "測試集 (20%)": n_test,
                    "FedAvg 權重佔比": f"{n_train / 4951 * 100:.1f}%",
                })
            st.markdown("#### 各分區資料切分與 FedAvg 聚合權重表")
            st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

    with tab2:
        col_sel, col_stat = st.columns([2, 3])
        with col_sel:
            sel_feat = st.selectbox(
                "選擇檢視之生理或衍生指標",
                options=ALL_FEATURES,
                index=2,
                format_func=lambda k: FEATURE_LABELS.get(k, k),
            )
        g1_mean = float(raw_df[raw_df["Group"] == "Group 1"][sel_feat].mean())
        g5_mean = float(raw_df[raw_df["Group"] == "Group 5"][sel_feat].mean())
        with col_stat:
            s1, s2 = st.columns(2)
            s1.metric("Group 1 平均值 (早期世代)", f"{g1_mean:.2f}")
            s2.metric("Group 5 平均值 (近期世代)", f"{g5_mean:.2f}", f"{g5_mean - g1_mean:+.2f} vs Group 1", delta_color="off")

        fig_box = px.box(
            raw_df,
            x="Group",
            y=sel_feat,
            color="Group",
            points="outliers",
            title=f"跨分區分佈盒鬚圖：{FEATURE_LABELS.get(sel_feat, sel_feat)}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_box.update_layout(height=420, showlegend=False, xaxis_title="", yaxis_title=sel_feat)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        imp_df, corr_g1, corr_g5 = compute_rf_and_corr()
        fig_imp = px.bar(
            imp_df,
            x="Feature",
            y="Importance",
            color="Group",
            barmode="group",
            title="早期世代 (Group 1) vs 近期世代 (Group 5) 特徵重要度位移",
            color_discrete_map={"Group 1": "#6366f1", "Group 5": "#10b981"},
        )
        fig_imp.update_layout(height=360, xaxis_title="", yaxis_title="Random Forest Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

        hc1, hc2 = st.columns(2)
        with hc1:
            fig_c1 = px.imshow(
                corr_g1.round(2),
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Group 1 特徵相關係數矩陣",
            )
            fig_c1.update_layout(height=400)
            st.plotly_chart(fig_c1, use_container_width=True)
        with hc2:
            fig_c5 = px.imshow(
                corr_g5.round(2),
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Group 5 特徵相關係數矩陣",
            )
            fig_c5.update_layout(height=400)
            st.plotly_chart(fig_c5, use_container_width=True)


# ==========================================
# 章節 2：聯邦學習架構與可重現性
# ==========================================
elif page == "2. 聯邦學習架構與可重現性":
    st.title("2. 系統架構：聯邦式學習 (FedAvg / FedProx) 與可重現性驗證")

    # 訓練範式精簡對照表
    st.subheader("四種機器學習訓練範式對照")
    paradigm_df = pd.DataFrame({
        "訓練範式": ["集中式訓練 (Centralized)", "單機獨立訓練 (Local-only)", "循序接力學習 (Sequential)", "水平聯邦學習 (FedAvg / FedProx - 本系統)"],
        "原始病歷位置": ["集中上傳單一伺服器", "保留於各醫院本地", "保留於各醫院本地", "保留於各醫院本地（僅傳權重）"],
        "隱私保護性": ["❌ 隱私外洩風險", "✅ 原始資料不出院", "✅ 原始資料不出院", "✅ 原始資料不出院"],
        "跨院知識共享": ["✅ 資料完整共享", "❌ 資料孤島", "⚠️ 遺忘早期分區", "✅ 全局加權聚合 + 近端抗漂移 + 本地微調"],
    })
    st.dataframe(paradigm_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("FedAvg 聚合與 FedProx 近端抗漂移目標函數")
        st.latex(r"\mathbf{w}^{t+1} = \sum_{k=1}^{5} \frac{n_k}{n} \, \mathbf{w}_k^{t+1}, \quad \min_{\mathbf{w}} \; \mathcal{L}_{\text{BCE}}(\mathbf{w}) + \frac{\mu}{2}\|\mathbf{w} - \mathbf{w}^t\|_2^2")
        st.markdown('<div class="tag-box">1. Server 廣播全域權重 wᵗ 與動態參數 (E, lr, μ, seed)</div>', unsafe_allow_html=True)
        st.markdown('<div class="tag-box">2. Client 重設種子 (seed + 1000·t + cid) 並計算近端正則化損失</div>', unsafe_allow_html=True)
        st.markdown('<div class="tag-box">3. Client 回傳更新權重 wₖᵗ⁺¹、樣本數 nₖ 與客戶端 ID (cid)</div>', unsafe_allow_html=True)
        st.markdown('<div class="tag-box">4. Server 依 int(cid) 嚴格排序後按樣本數比例加權平均</div>', unsafe_allow_html=True)

    with col_right:
        st.subheader("DiabetesNet 神經網路架構 (PyTorch)")
        arch_df = pd.DataFrame({
            "網路層 (Layer)": [
                "Input Layer",
                "Hidden Layer 1",
                "Hidden Layer 2",
                "Output Layer",
                "Loss / Optimizer",
            ],
            "結構與轉換規格": [
                "16 維標準化特徵 (13 原始生理 + 3 衍生指標)",
                "Linear(16 → 64) + BatchNorm1d(64) + ReLU + Dropout(0.2)",
                "Linear(64 → 32) + BatchNorm1d(32) + ReLU + Dropout(0.2)",
                "Linear(32 → 1) 輸出 Logit（推論時經 Sigmoid 轉機率）",
                "BCEWithLogitsLoss + FedProx Proximal Term + Adam (3,393 參數)",
            ],
        })
        st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("零誤差可重現性實測驗證 (check_repro.py 實測結果)")

    r_col1, r_col2 = st.columns(2)
    r_col1.metric("相同種子權重最大差 (Seed 0 vs 0)", "0.00e+00", "通過 (< 1e-6 標準)")
    r_col2.metric("不同種子權重最大差 (Seed 0 vs 1)", "1.65e+00", "確認隨機性有效 (> 1e-3)")

    p1 = "results_repro/seed0_run1/global_history.csv"
    p2 = "results_repro/seed0_run2/global_history.csv"
    p3 = "results_repro/seed1_run1/global_history.csv"
    if os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3):
        h1, h2, h3 = pd.read_csv(p1), pd.read_csv(p2), pd.read_csv(p3)
        repro_df = pd.DataFrame({
            "Round": h1["round"].astype(int),
            "Seed 0 (Run 1) AUC": h1["auc"].map(lambda x: f"{x:.6f}"),
            "Seed 0 (Run 2) AUC": h2["auc"].map(lambda x: f"{x:.6f}"),
            "同種子絕對誤差 |Run1 - Run2|": (h1["auc"] - h2["auc"]).abs().map(lambda x: f"{x:.2e}"),
            "Seed 1 (Run 1) AUC": h3["auc"].map(lambda x: f"{x:.6f}"),
            "異種子絕對差異 |Seed0 - Seed1|": (h1["auc"] - h3["auc"]).abs().map(lambda x: f"{x:.2e}"),
        })
        st.dataframe(repro_df, use_container_width=True, hide_index=True)


# ==========================================
# 章節 3：兩階段超參數調優 + 進階實驗 6.1 FedProx
# ==========================================
elif page == "3. 兩階段超參數調優":
    st.title("3. 超參數調優與抗漂移驗證：Stage A / Stage B 與 FedProx 實測")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("實驗規模", "100 場模擬", "Stage A (25) + B (25) + FedProx (50)")
    k2.metric("Stage A 最佳本地週期", "E* = 2", "E=10 發生 Client Drift")
    k3.metric("Stage B 最佳學習率", "lr* = 0.01", "搜尋區間正中央最佳解")
    k4.metric("FedProx 抗漂移增益", "+0.28 pp Test AUC", "E=10 下 μ=0.05 成功抑制漂移")

    st.markdown("---")
    metric_col, _ = st.columns([2, 3])
    with metric_col:
        curve_metric = st.segmented_control(
            "選擇曲線顯示指標",
            options=["auc", "f1", "acc", "loss"],
            default="auc",
            format_func=lambda m: {"auc": "Validation AUC", "f1": "Validation F1", "acc": "Validation Accuracy", "loss": "Validation Loss"}[m],
        )
        if curve_metric is None:
            curve_metric = "auc"

    tab_sa, tab_sb, tab_fp = st.tabs([
        "🔹 階段 A：本地週期掃描 (E ∈ {1, 2, 3, 5, 10})",
        "🔸 階段 B：學習率掃描 (lr ∈ {0.001 ~ 0.1})",
        "🔴 進階實驗 6.1：FedProx 近端正則化抗漂移驗證 (μ ∈ {0 ~ 0.1})",
    ])

    with tab_sa:
        st.subheader("階段 A (Stage A)：固定 lr = 0.01，尋找最佳本地訓練週期 E ∈ {1, 2, 3, 5, 10}")
        curves_a, table_a = load_stage_histories("stageA")

        if not curves_a.empty:
            ca1, ca2 = st.columns(2)
            with ca1:
                fig_a1 = go.Figure()
                for setting, grp in curves_a.groupby("setting", sort=False):
                    m = grp[f"{curve_metric}_mean"]
                    s = grp[f"{curve_metric}_std"]
                    fig_a1.add_trace(go.Scatter(x=grp["round"], y=m, mode="lines+markers", name=setting))
                    fig_a1.add_trace(go.Scatter(
                        x=pd.concat([grp["round"], grp["round"][::-1]]),
                        y=pd.concat([m + s, (m - s)[::-1]]),
                        fill="toself",
                        fillcolor="rgba(128,128,128,0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                fig_a1.update_layout(title="Stage A：指標 vs 通訊回合數 (Rounds)", xaxis_title="Communication Round", yaxis_title=curve_metric.upper(), height=380)
                st.plotly_chart(fig_a1, use_container_width=True)

            with ca2:
                fig_a2 = px.line(
                    curves_a,
                    x="total_epochs",
                    y=f"{curve_metric}_mean",
                    color="setting",
                    markers=True,
                    title="Stage A：指標 vs 總本地訓練週期 (Rounds × E)",
                    labels={"total_epochs": "Total Local Epochs (Rounds × E)", f"{curve_metric}_mean": curve_metric.upper()},
                )
                fig_a2.update_layout(height=380)
                st.plotly_chart(fig_a2, use_container_width=True)

            st.dataframe(table_a.drop(columns=["_auc_val"]), use_container_width=True, hide_index=True)

    with tab_sb:
        st.subheader("階段 B (Stage B)：固定 E* = 2，尋找最佳學習率 lr ∈ {0.001, 0.003, 0.01, 0.03, 0.1}")
        curves_b, table_b = load_stage_histories("stageB")

        if not curves_b.empty:
            cb1, cb2 = st.columns(2)
            with cb1:
                fig_b1 = go.Figure()
                for setting, grp in curves_b.groupby("setting", sort=False):
                    m = grp[f"{curve_metric}_mean"]
                    s = grp[f"{curve_metric}_std"]
                    fig_b1.add_trace(go.Scatter(x=grp["round"], y=m, mode="lines+markers", name=setting))
                    fig_b1.add_trace(go.Scatter(
                        x=pd.concat([grp["round"], grp["round"][::-1]]),
                        y=pd.concat([m + s, (m - s)[::-1]]),
                        fill="toself",
                        fillcolor="rgba(128,128,128,0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                fig_b1.update_layout(title="Stage B：指標 vs 通訊回合數 (Rounds)", xaxis_title="Communication Round", yaxis_title=curve_metric.upper(), height=380)
                st.plotly_chart(fig_b1, use_container_width=True)

            with cb2:
                fig_b2 = px.bar(
                    table_b,
                    x="參數組合",
                    y="_auc_val",
                    text="最終驗證 AUC",
                    color="參數組合",
                    title="Stage B：第 20 回合最終驗證 AUC 對比 (Interior Optimum @ lr=0.01)",
                )
                fig_b2.update_yaxes(range=[0.932, 0.938], title="Final Validation AUC")
                fig_b2.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_b2, use_container_width=True)

            st.dataframe(table_b.drop(columns=["_auc_val"]), use_container_width=True, hide_index=True)

    with tab_fp:
        st.subheader("進階實驗 6.1：FedProx 近端正則化抗客戶端漂移驗證 (FedAvg μ=0 vs FedProx μ>0)")
        fp_df = load_fedprox_curves()
        if not fp_df.empty:
            fp_ctrl1, fp_ctrl2 = st.columns([2, 3])
            with fp_ctrl1:
                sel_e = st.radio(
                    "選擇檢視之本地訓練週期情境",
                    options=[10, 2],
                    format_func=lambda e: "E = 10（高本地週期：檢驗抑制 Client Drift 能力）" if e == 10 else "E* = 2（最佳本地週期：檢驗極限效能提升）",
                    horizontal=True,
                )
            sub_fp = fp_df[fp_df["E"] == sel_e].copy()
            col_map = {"auc": "val_auc", "f1": "val_f1", "acc": "val_acc", "loss": "val_loss"}
            fp_col = col_map[curve_metric]

            fc1, fc2 = st.columns(2)
            with fc1:
                fig_fp1 = go.Figure()
                for setting, grp in sub_fp.groupby("setting", sort=False):
                    m = grp[f"{fp_col}_mean"]
                    fig_fp1.add_trace(go.Scatter(
                        x=grp["round"],
                        y=m,
                        mode="lines+markers",
                        name=setting,
                        line=dict(width=3 if "0.05" in setting or "0.01" in setting or "μ=0)" in setting else 1.8),
                    ))
                fig_fp1.update_layout(
                    title=f"E = {sel_e}：各近端係數 μ 之驗證集 {curve_metric.upper()} 收斂軌跡",
                    xaxis_title="Communication Round",
                    yaxis_title=f"Validation {curve_metric.upper()}",
                    height=380,
                )
                st.plotly_chart(fig_fp1, use_container_width=True)

            with fc2:
                r20 = sub_fp[sub_fp["round"] == 20].copy()
                fig_fp2 = px.bar(
                    r20,
                    x="setting",
                    y="test_auc_mean",
                    error_y="test_auc_std",
                    color="setting",
                    text=r20["test_auc_mean"].map(lambda v: f"{v:.4f}"),
                    title=f"E = {sel_e}：第 20 回合最終測試集 ROC-AUC 對比 (5 Seeds Mean ± Std)",
                )
                low_y = float(r20["test_auc_mean"].min()) - 0.004
                high_y = float(r20["test_auc_mean"].max()) + 0.003
                fig_fp2.update_yaxes(range=[low_y, high_y], title="Test ROC-AUC")
                fig_fp2.update_layout(height=380, xaxis_title="", showlegend=False)
                st.plotly_chart(fig_fp2, use_container_width=True)

            fp_table = []
            for _, row in r20.iterrows():
                fp_table.append({
                    "演算法與近端係數": row["setting"],
                    "本地週期 E": int(row["E"]),
                    "μ 值": row["mu"],
                    "最終驗證 AUC": f"{row['val_auc_mean']:.4f} ± {row['val_auc_std']:.4f}",
                    "最終驗證 F1": f"{row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f}",
                    "最終驗證 Loss": f"{row['val_loss_mean']:.4f} ± {row['val_loss_std']:.4f}",
                    "盲測 Test AUC": f"{row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f}",
                    "盲測 Test Loss": f"{row['test_loss_mean']:.4f} ± {row['test_loss_std']:.4f}",
                })
            st.dataframe(pd.DataFrame(fp_table), use_container_width=True, hide_index=True)


# ==========================================
# 章節 4：最終盲測效能對決 + 進階實驗 6.2 個人化微調最佳化
# ==========================================
elif page == "4. 最終盲測效能對決":
    st.title("4. 最終效能評估：全新種子 (Seeds 10~14) 測試集三方盲測與個人化調優")

    df_base = pd.read_csv("results/final/final_test_results.csv")
    df_pers = load_pers_tuning()

    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.6, 2.2, 1.4])
    with c_ctrl1:
        metric_choice = st.selectbox(
            "選擇比較指標",
            options=["f1", "auc", "accuracy", "loss"],
            format_func=lambda x: {
                "f1": "F1-Score",
                "auc": "ROC-AUC",
                "accuracy": "Accuracy",
                "loss": "BCE Loss (越低越好)",
            }[x],
        )
    with c_ctrl2:
        pers_options = [
            "Moderate (E=2, lr=0.001)",
            "Mild (E=1, lr=0.001)",
            "Gentle (E=1, lr=0.0005)",
            "Original (E=3, lr=0.005)",
        ]
        pers_choice = st.selectbox(
            "個人化微調強度設定 (進階實驗 6.2)",
            options=pers_options,
            index=0,
            format_func=lambda s: {
                "Moderate (E=2, lr=0.001)": "✨ 溫和微調 (E=2, lr=0.001) [最佳推薦：消除個人化漂移]",
                "Mild (E=1, lr=0.001)": "🔹 輕量微調 (E=1, lr=0.001)",
                "Gentle (E=1, lr=0.0005)": "🌱 極輕微調 (E=1, lr=0.0005)",
                "Original (E=3, lr=0.005)": "⚠️ 原始強烈微調 (E=3, lr=0.005) [Group 5 發生過度微調]",
            }[s],
        )
    with c_ctrl3:
        y_zoom = st.toggle("放大 Y 軸凸顯組間差異", value=True)

    # 依據所選之個人化微調強度動態組裝三方比較資料表
    df_res = df_base[df_base["model_type"].isin(["Local-only", "Global-only"])].copy()
    if not df_pers.empty and pers_choice != "Original (E=3, lr=0.005)":
        sub_p = df_pers[df_pers["setting"] == pers_choice].copy()
        sub_p_mapped = pd.DataFrame({
            "seed": sub_p["seed"].astype(int),
            "client": sub_p["client"],
            "model_type": "Personalized",
            "loss": sub_p["test_loss"],
            "accuracy": sub_p["test_acc"],
            "f1": sub_p["test_f1"],
            "auc": sub_p["test_auc"],
        })
        df_res = pd.concat([df_res, sub_p_mapped], ignore_index=True)
    else:
        df_res = df_base.copy()

    metric_label_map = {
        "f1": "F1-Score",
        "auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "loss": "BCE Loss",
    }
    cur_label = metric_label_map[metric_choice]

    loc_df = df_res[df_res["model_type"] == "Local-only"]
    glb_df = df_res[df_res["model_type"] == "Global-only"]
    per_df = df_res[df_res["model_type"] == "Personalized"]

    loc_mean, loc_std = float(loc_df[metric_choice].mean()), float(np.std(loc_df[metric_choice], ddof=0))
    glb_mean, glb_std = float(glb_df[metric_choice].mean()), float(np.std(glb_df[metric_choice], ddof=0))
    per_mean, per_std = float(per_df[metric_choice].mean()), float(np.std(per_df[metric_choice], ddof=0))

    if metric_choice == "loss":
        glb_delta = f"{glb_mean - loc_mean:+.4f} vs Local Only"
        per_delta = f"{per_mean - loc_mean:+.4f} vs Local Only"
        d_color = "inverse"
    else:
        glb_delta = f"{(glb_mean - loc_mean) * 100:+.2f} pp vs Local Only"
        per_delta = f"{(per_mean - loc_mean) * 100:+.2f} pp vs Local Only"
        d_color = "normal"

    # 頂部三種模型數值標籤 (Local Only、Global Only、Personalized FL)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(f"Local Only｜平均 {cur_label}", f"{loc_mean:.4f} ± {loc_std:.4f}", "單機基準線 (Baseline)", delta_color="off")
        st.caption(f"AUC: {loc_df['auc'].mean():.4f} ｜ F1: {loc_df['f1'].mean():.4f} ｜ Acc: {loc_df['accuracy'].mean():.4f} ｜ Loss: {loc_df['loss'].mean():.4f}")
    with m2:
        st.metric(f"Global Only｜平均 {cur_label}", f"{glb_mean:.4f} ± {glb_std:.4f}", glb_delta, delta_color=d_color)
        st.caption(f"AUC: {glb_df['auc'].mean():.4f} ｜ F1: {glb_df['f1'].mean():.4f} ｜ Acc: {glb_df['accuracy'].mean():.4f} ｜ Loss: {glb_df['loss'].mean():.4f}")
    with m3:
        st.metric(f"Personalized FL｜平均 {cur_label}", f"{per_mean:.4f} ± {per_std:.4f}", per_delta, delta_color=d_color)
        st.caption(f"AUC: {per_df['auc'].mean():.4f} ｜ F1: {per_df['f1'].mean():.4f} ｜ Acc: {per_df['accuracy'].mean():.4f} ｜ Loss: {per_df['loss'].mean():.4f}")

    st.markdown("---")

    tab_main, tab_tune = st.tabs([
        "🏆 三方盲測對決 (Local-only vs Global-only vs Personalized)",
        "🔬 進階實驗 6.2：個人化微調強度最佳化 (消除 Group 5 個人化漂移)",
    ])

    with tab_main:
        summary_plot = (
            df_res.groupby(["client", "model_type"], sort=False)[metric_choice]
            .agg(mean="mean", std=lambda x: np.std(x, ddof=0))
            .reset_index()
        )

        fig_bar = px.bar(
            summary_plot,
            x="client",
            y="mean",
            error_y="std",
            color="model_type",
            barmode="group",
            text=summary_plot["mean"].map(lambda v: f"{v:.4f}"),
            title=f"各分區測試集 {metric_choice.upper()} 對比 — 個人化設定：{pers_choice} (Seeds 10~14 Mean ± Std)",
            color_discrete_map={
                "Local-only": "#f97316",
                "Global-only": "#10b981",
                "Personalized": "#6366f1",
            },
        )
        if y_zoom and metric_choice != "loss":
            low = max(0.74, float(summary_plot["mean"].min()) - 0.03)
            high = min(0.96, float(summary_plot["mean"].max()) + 0.02)
            fig_bar.update_yaxes(range=[low, high])
        fig_bar.update_layout(height=420, xaxis_title="", yaxis_title=metric_choice.upper(), legend_title="模型架構")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        t_col1, t_col2 = st.columns([3, 2])
        with t_col1:
            st.markdown("#### 表一：各分區測試集表現 (Mean ± Std)")
            t1 = (
                df_res.groupby(["client", "model_type"], sort=False)[["auc", "f1", "accuracy", "loss"]]
                .agg(lambda x: f"{np.mean(x):.4f} ± {np.std(x, ddof=0):.4f}")
                .reset_index()
                .rename(columns={"client": "分區", "model_type": "模型架構", "auc": "ROC-AUC", "f1": "F1-Score", "accuracy": "Accuracy", "loss": "BCE Loss"})
            )
            st.dataframe(t1, use_container_width=True, hide_index=True)

        with t_col2:
            st.markdown("#### 表二：三種模型架構全域總平均表現 (Mean ± Std)")
            t2 = (
                df_res.groupby("model_type", sort=False)[["auc", "f1", "accuracy", "loss"]]
                .agg(lambda x: f"{np.mean(x):.4f} ± {np.std(x, ddof=0):.4f}")
                .reset_index()
                .rename(columns={"model_type": "模型架構", "auc": "平均 AUC", "f1": "平均 F1", "accuracy": "平均 Acc", "loss": "平均 Loss"})
            )
            st.dataframe(t2, use_container_width=True, hide_index=True)

    with tab_tune:
        if not df_pers.empty:
            p_col_map = {"f1": "test_f1", "auc": "test_auc", "accuracy": "test_acc", "loss": "test_loss"}
            p_col = p_col_map[metric_choice]

            tune_grp = (
                df_pers.groupby(["client", "setting"], sort=False)[p_col]
                .agg(mean="mean", std=lambda x: np.std(x, ddof=0))
                .reset_index()
            )
            fig_tune = px.bar(
                tune_grp,
                x="client",
                y="mean",
                error_y="std",
                color="setting",
                barmode="group",
                text=tune_grp["mean"].map(lambda v: f"{v:.4f}"),
                title=f"不同個人化微調強度於各分區之測試集 {cur_label} 對比 (溫和微調成功消除 Group 5 退化)",
                color_discrete_sequence=["#10b981", "#38bdf8", "#6366f1", "#4f46e5", "#ef4444"],
            )
            if y_zoom and metric_choice != "loss":
                low_t = max(0.75, float(tune_grp["mean"].min()) - 0.025)
                high_t = min(0.96, float(tune_grp["mean"].max()) + 0.015)
                fig_tune.update_yaxes(range=[low_t, high_t])
            fig_tune.update_layout(height=420, xaxis_title="", yaxis_title=cur_label, legend_title="微調強度設定")
            st.plotly_chart(fig_tune, use_container_width=True)

            # 彙整比較表
            tune_summary = []
            for s_name, grp in df_pers.groupby("setting", sort=False):
                g1_f1 = grp[grp["client"] == "Group1"]["test_f1"].mean()
                g5_f1 = grp[grp["client"] == "Group5"]["test_f1"].mean()
                g5_auc = grp[grp["client"] == "Group5"]["test_auc"].mean()
                tune_summary.append({
                    "個人化微調設定": s_name,
                    "微調週期 E_p": int(grp["pers_epochs"].iloc[0]),
                    "微調學習率 lr_p": float(grp["pers_lr"].iloc[0]),
                    "全區平均 Test AUC": f"{grp['test_auc'].mean():.4f} ± {grp['test_auc'].std(ddof=0):.4f}",
                    "全區平均 Test F1": f"{grp['test_f1'].mean():.4f} ± {grp['test_f1'].std(ddof=0):.4f}",
                    "全區平均 Test Acc": f"{grp['test_acc'].mean():.4f} ± {grp['test_acc'].std(ddof=0):.4f}",
                    "全區平均 Test Loss": f"{grp['test_loss'].mean():.4f}",
                    "Group 1 F1 (小樣本增益)": f"{g1_f1:.4f}",
                    "Group 5 AUC / F1 (大樣本防漂移)": f"{g5_auc:.4f} / {g5_f1:.4f}",
                })
            st.markdown("#### 個人化微調強度全維度對照表 (Seeds 10~14)")
            st.dataframe(pd.DataFrame(tune_summary), use_container_width=True, hide_index=True)


# ==========================================
# 章節 5：Live Demo 臨床即時推論
# ==========================================
elif page == "5. Live Demo 臨床即時推論":
    st.title("5. Live Demo：真實聯邦全域模型 (DiabetesNet) 即時推論")
    st.caption("載入 `results/final/E2_lr0.01/seed10~14/global_model.npz` 真實權重與 Group 1 `StandardScaler` 進行 16 維前向傳播（僅供學術展示，非醫療診斷）")

    scaler, healthy_medians = get_base_scaler_and_medians()
    raw_df = load_raw_partitions()

    # 預設快速範本按鈕
    PRE_HEALTHY = {
        "RIDAGEYR": 38, "RIAGENDR": 2.0, "LBXGH": 5.2, "LBXGLU": 89.0,
        "LBXHGB": 13.8, "LBXRBCSI": 4.60, "LBXHCT": 40.8, "LBDHDD": 58.0,
        "LBDLDL": 96.0, "URXUMA": 6.5, "LBXSCR": 0.78, "LBXSASSI": 19.0, "LBXSGTSI": 16.0,
    }
    PRE_BORDER = {
        "RIDAGEYR": 56, "RIAGENDR": 1.0, "LBXGH": 6.1, "LBXGLU": 116.0,
        "LBXHGB": 14.5, "LBXRBCSI": 4.85, "LBXHCT": 43.0, "LBDHDD": 44.0,
        "LBDLDL": 128.0, "URXUMA": 22.0, "LBXSCR": 0.95, "LBXSASSI": 25.0, "LBXSGTSI": 34.0,
    }
    PRE_HIGH = {
        "RIDAGEYR": 67, "RIAGENDR": 1.0, "LBXGH": 7.8, "LBXGLU": 168.0,
        "LBXHGB": 14.1, "LBXRBCSI": 4.72, "LBXHCT": 41.9, "LBDHDD": 38.0,
        "LBDLDL": 145.0, "URXUMA": 65.0, "LBXSCR": 1.18, "LBXSASSI": 31.0, "LBXSGTSI": 48.0,
    }

    if "demo_vals" not in st.session_state:
        st.session_state.demo_vals = dict(PRE_BORDER)
        st.session_state.demo_meta = "預設範例：糖尿病前期邊緣案例"

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🟢 載入健康標準案例", use_container_width=True):
        st.session_state.demo_vals = dict(PRE_HEALTHY)
        st.session_state.demo_meta = "預設範例：健康標準案例"
    if b2.button("🟡 載入糖尿病前期案例", use_container_width=True):
        st.session_state.demo_vals = dict(PRE_BORDER)
        st.session_state.demo_meta = "預設範例：糖尿病前期邊緣案例"
    if b3.button("🔴 載入高風險典型案例", use_container_width=True):
        st.session_state.demo_vals = dict(PRE_HIGH)
        st.session_state.demo_meta = "預設範例：高風險典型案例"
    if b4.button("🎲 隨機抽樣真實 NHANES 案例", use_container_width=True):
        sample_row = raw_df.sample(1).iloc[0]
        st.session_state.demo_vals = {k: float(sample_row[k]) for k in RAW_FEATURES}
        true_lbl = "高風險 (DIQ010=1.0)" if sample_row["DIQ010"] == 1.0 else "健康 (DIQ010=0.0)"
        st.session_state.demo_meta = f"真實抽樣個案｜來源分區：{sample_row['Group']}｜真實診斷：{true_lbl}"

    dv = st.session_state.demo_vals
    st.markdown(f'<div class="tag-box">📌 當前檢視個案：{st.session_state.get("demo_meta", "")}</div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### ① 核心血糖與人口統計")
        age = st.slider("年齡 (RIDAGEYR, 歲)", 12, 85, int(dv["RIDAGEYR"]))
        sex_val = st.segmented_control("生理性別 (RIAGENDR)", options=[1.0, 2.0], default=float(dv["RIAGENDR"]), format_func=lambda x: "男性 (1.0)" if x == 1.0 else "女性 (2.0)")
        if sex_val is None:
            sex_val = float(dv["RIAGENDR"])
        hba1c = st.slider("糖化血色素 (LBXGH, %)", 3.5, 15.0, float(dv["LBXGH"]), 0.1)
        glu = st.slider("空腹血糖 (LBXGLU, mg/dL)", 50.0, 350.0, float(dv["LBXGLU"]), 1.0)

    with c2:
        st.markdown("#### ② 血脂與腎臟指標")
        ldl = st.slider("低密度膽固醇 LDL (LBDLDL, mg/dL)", 20.0, 260.0, float(dv["LBDLDL"]), 1.0)
        hdl = st.slider("高密度膽固醇 HDL (LBDHDD, mg/dL)", 17.0, 120.0, float(dv["LBDHDD"]), 1.0)
        urxuma = st.slider("尿液微量白蛋白 (URXUMA, μg/mL)", 0.5, 300.0, float(min(dv["URXUMA"], 300.0)), 0.5)
        scr = st.slider("血清肌酸酐 (LBXSCR, mg/dL)", 0.3, 4.0, float(min(dv["LBXSCR"], 4.0)), 0.05)

    with c3:
        st.markdown("#### ③ 血液常規與肝指數")
        hgb = st.slider("血紅素 (LBXHGB, g/dL)", 8.0, 19.0, float(dv["LBXHGB"]), 0.1)
        rbc = st.slider("紅血球數 (LBXRBCSI, 10⁶/μL)", 2.5, 7.5, float(dv["LBXRBCSI"]), 0.05)
        hct = st.slider("血球容積比 (LBXHCT, %)", 24.0, 55.0, float(dv["LBXHCT"]), 0.5)
        ast = st.slider("AST 肝指數 (LBXSASSI, U/L)", 6.0, 120.0, float(min(dv["LBXSASSI"], 120.0)), 1.0)
        ggt = st.slider("GGT 肝指數 (LBXSGTSI, U/L)", 4.0, 200.0, float(min(dv["LBXSGTSI"], 200.0)), 1.0)

    current_raw = {
        "RIDAGEYR": float(age),
        "RIAGENDR": float(sex_val),
        "LBXGH": float(hba1c),
        "LBXHGB": float(hgb),
        "LBXGLU": float(glu),
        "LBXRBCSI": float(rbc),
        "URXUMA": float(urxuma),
        "LBDHDD": float(hdl),
        "LBDLDL": float(ldl),
        "LBXSASSI": float(ast),
        "LBXSGTSI": float(ggt),
        "LBXSCR": float(scr),
        "LBXHCT": float(hct),
    }

    # 衍生特徵即時卡片
    gh_glu = current_raw["LBXGH"] * current_raw["LBXGLU"]
    uacr = current_raw["URXUMA"] / max(current_raw["LBXSCR"], 0.01)
    lipid_r = current_raw["LBDLDL"] / max(current_raw["LBDHDD"], 1.0)

    d1, d2, d3 = st.columns(3)
    d1.metric("衍生特徵 ①：GH_GLU_Interact (LBXGH × LBXGLU)", f"{gh_glu:.1f}", "高風險 (> 700)" if gh_glu > 700 else "正常區間", delta_color="inverse")
    d2.metric("衍生特徵 ②：UACR_Proxy (URXUMA / LBXSCR)", f"{uacr:.2f}", "微量白蛋白尿 (> 30)" if uacr > 30 else "正常區間", delta_color="inverse")
    d3.metric("衍生特徵 ③：Lipid_Ratio (LBDLDL / LBDHDD)", f"{lipid_r:.2f}", "血脂比偏高 (> 3.0)" if lipid_r > 3.0 else "正常區間", delta_color="inverse")

    st.markdown("---")

    # 推論模型選擇器：支援 Global-only 全域模型 與 Personalized 各分區個人化微調模型
    m_sel_col1, m_sel_col2 = st.columns([2, 3])
    with m_sel_col1:
        demo_model_mode = st.selectbox(
            "選擇即時推論模型架構",
            options=["Global-only", "Group 1", "Group 2", "Group 3", "Group 4", "Group 5"],
            index=0,
            format_func=lambda m: (
                "🌐 Global-only（聯邦全域模型 E*=2, lr*=0.01）"
                if m == "Global-only"
                else f"🏥 Personalized FL — {m} 個人化微調模型 (E_p=2, lr_p=0.001)"
            ),
        )

    pers_models_dict = get_personalized_weights_dict(pers_epochs=2, pers_lr=0.001)
    x_scaled = build_feature_vector(current_raw, scaler)

    seed_probs = []
    for s in [10, 11, 12, 13, 14]:
        if demo_model_mode == "Global-only":
            npz_p = f"results/final/E2_lr0.01/seed{s}/global_model.npz"
            if os.path.exists(npz_p):
                seed_probs.append({"Seed": f"Seed {s}", "Prob": predict_with_npz(x_scaled, npz_p) * 100.0})
        else:
            w_dict = pers_models_dict.get(demo_model_mode, {}).get(s)
            if w_dict is not None:
                seed_probs.append({"Seed": f"Seed {s}", "Prob": predict_with_weights(x_scaled, w_dict) * 100.0})

    mean_prob = float(np.mean([r["Prob"] for r in seed_probs]))
    std_prob = float(np.std([r["Prob"] for r in seed_probs], ddof=0))

    # 同步計算 Global-only 與 5 家醫院 Personalized 模型於此病患之預測機率對比
    hosp_compare_rows = []
    glb_vals = [
        predict_with_npz(x_scaled, f"results/final/E2_lr0.01/seed{s}/global_model.npz") * 100.0
        for s in [10, 11, 12, 13, 14]
        if os.path.exists(f"results/final/E2_lr0.01/seed{s}/global_model.npz")
    ]
    hosp_compare_rows.append({"模型": "Global-only", "平均風險 (%)": float(np.mean(glb_vals)), "類型": "Global"})
    for g_name in ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5"]:
        p_vals = [
            predict_with_weights(x_scaled, pers_models_dict[g_name][s]) * 100.0
            for s in [10, 11, 12, 13, 14]
            if s in pers_models_dict.get(g_name, {})
        ]
        hosp_compare_rows.append({"模型": f"Pers ({g_name})", "平均風險 (%)": float(np.mean(p_vals)), "類型": "Personalized"})

    # 計算單一特徵邊際貢獻（將單一特徵替換為健康中位數後之預測差值）
    contrib_rows = []
    if demo_model_mode == "Global-only":
        ref_weights = dict(np.load("results/final/E2_lr0.01/seed10/global_model.npz"))
    else:
        ref_weights = pers_models_dict[demo_model_mode][10]
    base_p = predict_with_weights(x_scaled, ref_weights) * 100.0
    for feat in RAW_FEATURES:
        alt_raw = dict(current_raw)
        alt_raw[feat] = float(healthy_medians[feat])
        alt_p = predict_with_weights(build_feature_vector(alt_raw, scaler), ref_weights) * 100.0
        contrib_rows.append({
            "特徵": FEATURE_LABELS.get(feat, feat),
            "風險影響 (百分點)": base_p - alt_p,
        })
    contrib_df = pd.DataFrame(contrib_rows).sort_values("風險影響 (百分點)", key=lambda s: s.abs(), ascending=True).tail(8)

    res_left, res_mid, res_right = st.columns([1.3, 1.2, 1.5])

    with res_left:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_prob,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": f"{demo_model_mode} 預測風險機率 (±{std_prob:.2f}%)", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if mean_prob >= 50 else "#10b981"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [50, 75], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 4}, "thickness": 0.8, "value": 50},
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if mean_prob >= 75:
            st.error(f"🔴 **高風險族群 (≥ 75%)**：預測機率 {mean_prob:.2f}%（跨越 50% 決策門檻）")
        elif mean_prob >= 50:
            st.warning(f"🟡 **中度 / 邊緣風險 (50% ~ 75%)**：預測機率 {mean_prob:.2f}%（跨越 50% 決策門檻）")
        else:
            st.success(f"🟢 **低風險健康族群 (< 50%)**：預測機率 {mean_prob:.2f}%（低於 50% 決策門檻）")

    with res_mid:
        hosp_df = pd.DataFrame(hosp_compare_rows)
        fig_hosp = px.bar(
            hosp_df,
            x="模型",
            y="平均風險 (%)",
            color="類型",
            text=hosp_df["平均風險 (%)"].map(lambda v: f"{v:.1f}%"),
            title="Global vs 各院 Personalized 模型預測對比",
            color_discrete_map={"Global": "#10b981", "Personalized": "#6366f1"},
        )
        fig_hosp.add_hline(y=50, line_dash="dash", line_color="#ef4444")
        fig_hosp.update_yaxes(range=[0, 105], title="預測風險機率 (%)")
        fig_hosp.update_layout(height=360, xaxis_title="", showlegend=False)
        st.plotly_chart(fig_hosp, use_container_width=True)

    with res_right:
        contrib_df["方向"] = np.where(contrib_df["風險影響 (百分點)"] >= 0, "推升風險", "降低風險")
        fig_contrib = px.bar(
            contrib_df,
            x="風險影響 (百分點)",
            y="特徵",
            orientation="h",
            color="方向",
            text=contrib_df["風險影響 (百分點)"].map(lambda v: f"{v:+.1f} pp"),
            title=f"{demo_model_mode} 個別特徵影響拆解 (相對於健康中位數)",
            color_discrete_map={"推升風險": "#ef4444", "降低風險": "#10b981"},
        )
        fig_contrib.update_layout(height=360, xaxis_title="對預測機率之邊際影響 (百分點 pp)", yaxis_title="", showlegend=False)
        st.plotly_chart(fig_contrib, use_container_width=True)
