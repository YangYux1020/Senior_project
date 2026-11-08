import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
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


def predict_with_npz(x_scaled: np.ndarray, npz_path: str) -> float:
    """使用真實訓練完成之 DiabetesNet (.npz) 權重進行前向傳播推論"""
    d = np.load(npz_path)
    x = x_scaled.astype(np.float32) @ d["arr_0"].T + d["arr_1"]
    x = (x - d["arr_4"]) / np.sqrt(d["arr_5"] + 1e-5) * d["arr_2"] + d["arr_3"]
    x = np.maximum(x, 0.0)
    x = x @ d["arr_7"].T + d["arr_8"]
    x = (x - d["arr_11"]) / np.sqrt(d["arr_12"] + 1e-5) * d["arr_9"] + d["arr_10"]
    x = np.maximum(x, 0.0)
    logit = float((x @ d["arr_14"].T + d["arr_15"]).reshape(-1)[0])
    return float(1.0 / (1.0 + np.exp(-logit)))


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
    - **聚合演算法**：`FedAvg` (20 Rounds)
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
        groups_data = [g[sel_feat].values for _, g in raw_df.groupby("Group")]
        f_stat, p_val = stats.f_oneway(*groups_data)
        ks_stat, ks_p = stats.ks_2samp(
            raw_df[raw_df["Group"] == "Group 1"][sel_feat],
            raw_df[raw_df["Group"] == "Group 5"][sel_feat],
        )
        with col_stat:
            s1, s2 = st.columns(2)
            s1.metric("跨五區 ANOVA 檢定", f"F = {f_stat:.2f}", f"p = {p_val:.2e}")
            s2.metric("Group 1 vs Group 5 KS 檢定", f"D = {ks_stat:.3f}", f"p = {ks_p:.2e}")

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
    st.title("2. 系統架構：聯邦式學習 (Federated Learning) 與可重現性驗證")

    # 訓練範式精簡對照表
    st.subheader("四種機器學習訓練範式對照")
    paradigm_df = pd.DataFrame({
        "訓練範式": ["集中式訓練 (Centralized)", "單機獨立訓練 (Local-only)", "循序接力學習 (Sequential)", "水平聯邦學習 (FedAvg - 本系統)"],
        "原始病歷位置": ["集中上傳單一伺服器", "保留於各醫院本地", "保留於各醫院本地", "保留於各醫院本地（僅傳權重）"],
        "隱私保護性": ["❌ 隱私外洩風險", "✅ 原始資料不出院", "✅ 原始資料不出院", "✅ 原始資料不出院"],
        "跨院知識共享": ["✅ 資料完整共享", "❌ 資料孤島", "⚠️ 遺忘早期分區", "✅ 全局加權聚合 + 本地個人化微調"],
    })
    st.dataframe(paradigm_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("FedAvg 聚合流程與樣本數加權")
        st.latex(r"\mathbf{w}^{t+1} = \sum_{k=1}^{5} \frac{n_k}{n} \, \mathbf{w}_k^{t+1}, \quad n = \sum_{k=1}^{5} n_k = 4,951")
        st.markdown('<div class="tag-box">1. Server 廣播全域權重 wᵗ 與訓練參數 (E, lr, seed)</div>', unsafe_allow_html=True)
        st.markdown('<div class="tag-box">2. Client 動態重設種子 (seed + 1000·t + cid) 並執行本地訓練</div>', unsafe_allow_html=True)
        st.markdown('<div class="tag-box">3. Client 回傳更新權重 wₖᵗ⁺¹、樣本數 nₖ 與客戶端 ID</div>', unsafe_allow_html=True)
        st.markdown('<div class="tag-box">4. Server 依 cid 排序後按樣本數比例加權平均</div>', unsafe_allow_html=True)

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
                "BCEWithLogitsLoss + Adam Optimizer (可訓練參數：3,393 個)",
            ],
        })
        st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("零誤差可重現性實測驗證 (check_repro.py 實測結果)")

    r_col1, r_col2= st.columns(2)
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
# 章節 3：兩階段超參數調優
# ==========================================
elif page == "3. 兩階段超參數調優":
    st.title("3. 訓練與收斂歷程：驗證集兩階段超參數掃描 (Seeds 0~4)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("實驗規模", "50 場模擬", "Stage A (25) + Stage B (25)")
    k2.metric("評估資料集", "僅用驗證集 (12%)", "測試集全程封存")
    k3.metric("Stage A 選定本地週期", "E* = 2", "Val AUC 0.9361 ± 0.0008")
    k4.metric("Stage B 選定學習率", "lr* = 0.01", "搜尋區間正中央最佳解")

    st.markdown("---")
    metric_col, _ = st.columns([2, 3])
    with metric_col:
        curve_metric = st.segmented_control(
            "選擇曲線顯示指標",
            options=["auc", "f1", "acc", "loss"],
            default="auc",
            format_func=lambda m: {"auc": "Validation AUC", "f1": "Validation F1", "acc": "Validation Accuracy", "loss": "Validation Loss"}[m],
        )

    # 階段 A
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

    st.markdown("---")

    # 階段 B
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


# ==========================================
# 章節 4：最終盲測效能對決
# ==========================================
elif page == "4. 最終盲測效能對決":
    st.title("4. 最終效能評估：全新種子 (Seeds 10~14) 測試集三方盲測")

    df_res = pd.read_csv("results/final/final_test_results.csv")

    # 頂部統計指標與 p-value
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Global-only 平均 AUC", "0.9259 ± 0.0068", "+0.75 pp vs Local (p = 0.0002)")
    m2.metric("Personalized 平均 F1", "0.8437 ± 0.0184", "+1.09 pp vs Local (p = 0.0022)")
    m3.metric("Personalized 平均準確率", "0.8528 ± 0.0098", "+1.11 pp vs Local (p = 0.0010)")
    m4.metric("Global-only 平均 Loss", "0.3499 ± 0.0181", "-0.0255 vs Local (p = 0.0001)")

    st.markdown("---")

    c_ctrl1, c_ctrl2 = st.columns([2, 3])
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
        y_zoom = st.toggle("放大 Y 軸凸顯組間差異", value=True)

    summary_plot = (
        df_res.groupby(["client", "model_type"], sort=False)[metric_choice]
        .agg(mean="mean", std=lambda x: np.std(x, ddof=0))
        .reset_index()
    )

    ch_left, ch_right = st.columns([3, 2])
    with ch_left:
        fig_bar = px.bar(
            summary_plot,
            x="client",
            y="mean",
            error_y="std",
            color="model_type",
            barmode="group",
            text=summary_plot["mean"].map(lambda v: f"{v:.4f}"),
            title=f"各分區測試集 {metric_choice.upper()} 對比 (Seeds 10~14 Mean ± Std)",
            color_discrete_map={
                "Local-only": "#f97316",
                "Global-only": "#10b981",
                "Personalized": "#6366f1",
            },
        )
        if y_zoom and metric_choice != "loss":
            low = max(0.75, float(summary_plot["mean"].min()) - 0.03)
            high = min(0.96, float(summary_plot["mean"].max()) + 0.02)
            fig_bar.update_yaxes(range=[low, high])
        fig_bar.update_layout(height=420, xaxis_title="", yaxis_title=metric_choice.upper(), legend_title="模型架構")
        st.plotly_chart(fig_bar, use_container_width=True)

    with ch_right:
        piv_g = summary_plot.pivot(index="client", columns="model_type", values="mean").reset_index()
        piv_g["Global vs Local (pp)"] = (piv_g["Global-only"] - piv_g["Local-only"]) * 100
        piv_g["Personalized vs Local (pp)"] = (piv_g["Personalized"] - piv_g["Local-only"]) * 100
        diff_melt = piv_g.melt(
            id_vars="client",
            value_vars=["Global vs Local (pp)", "Personalized vs Local (pp)"],
            var_name="比較組別",
            value_name="提升百分點 (pp)",
        )
        fig_diff = px.bar(
            diff_melt,
            x="client",
            y="提升百分點 (pp)",
            color="比較組別",
            barmode="group",
            text=diff_melt["提升百分點 (pp)"].map(lambda v: f"{v:+.2f}"),
            title=f"相較於 Local-only 之 {metric_choice.upper()} 增幅 (百分點 pp)",
            color_discrete_map={"Global vs Local (pp)": "#10b981", "Personalized vs Local (pp)": "#6366f1"},
        )
        fig_diff.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_diff.update_layout(height=420, xaxis_title="")
        st.plotly_chart(fig_diff, use_container_width=True)

    # 三大發現摘要標籤
    f1_c, f2_c, f3_c = st.columns(3)
    with f1_c:
        st.markdown('<div class="tag-box">① Group 2 顯著突破：F1 +3.14 pp / Acc +3.19 pp</div>', unsafe_allow_html=True)
    with f2_c:
        st.markdown('<div class="tag-box">② Group 1 決策門檻校準：微調較 Global 提升 +2.58 pp F1</div>', unsafe_allow_html=True)
    with f3_c:
        st.markdown('<div class="tag-box">③ Group 4 / 5 全域模型達頂峰：驗證個人化漂移效應</div>', unsafe_allow_html=True)

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
        st.markdown("#### 表二：全域總平均與成對統計檢定 (n = 25 對)")
        t2 = (
            df_res.groupby("model_type", sort=False)[["auc", "f1", "accuracy", "loss"]]
            .agg(lambda x: f"{np.mean(x):.4f} ± {np.std(x, ddof=0):.4f}")
            .reset_index()
            .rename(columns={"model_type": "模型架構", "auc": "平均 AUC", "f1": "平均 F1", "accuracy": "平均 Acc", "loss": "平均 Loss"})
        )
        st.dataframe(t2, use_container_width=True, hide_index=True)

        piv_stat = df_res.pivot_table(index=["seed", "client"], columns="model_type", values=["auc", "f1", "accuracy"])
        stat_rows = []
        for a, b in [("Global-only", "Local-only"), ("Personalized", "Local-only"), ("Personalized", "Global-only")]:
            stat_rows.append({
                "成對檢定組別": f"{a} vs {b}",
                "ΔAUC (p-val)": f"{(piv_stat[('auc', a)] - piv_stat[('auc', b)]).mean():+.4f} (p={stats.ttest_rel(piv_stat[('auc', a)], piv_stat[('auc', b)]).pvalue:.4f})",
                "ΔF1 (p-val)": f"{(piv_stat[('f1', a)] - piv_stat[('f1', b)]).mean():+.4f} (p={stats.ttest_rel(piv_stat[('f1', a)], piv_stat[('f1', b)]).pvalue:.4f})",
                "ΔAcc (p-val)": f"{(piv_stat[('accuracy', a)] - piv_stat[('accuracy', b)]).mean():+.4f} (p={stats.ttest_rel(piv_stat[('accuracy', a)], piv_stat[('accuracy', b)]).pvalue:.4f})",
            })
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)


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

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🟢 載入健康標準案例", use_container_width=True):
        st.session_state.demo_vals = dict(PRE_HEALTHY)
    if b2.button("🟡 載入糖尿病前期案例", use_container_width=True):
        st.session_state.demo_vals = dict(PRE_BORDER)
    if b3.button("🔴 載入高風險典型案例", use_container_width=True):
        st.session_state.demo_vals = dict(PRE_HIGH)
    if b4.button("🎲 隨機抽樣真實 NHANES 案例", use_container_width=True):
        sample_row = raw_df.sample(1).iloc[0]
        st.session_state.demo_vals = {k: float(sample_row[k]) for k in RAW_FEATURES}

    dv = st.session_state.demo_vals

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
        urxuma = st.slider("尿液微量白蛋白 (URXUMA, μg/mL)", 0.5, 300.0, float( min(dv["URXUMA"], 300.0) ), 0.5)
        scr = st.slider("血清肌酸酐 (LBXSCR, mg/dL)", 0.3, 4.0, float( min(dv["LBXSCR"], 4.0) ), 0.05)

    with c3:
        st.markdown("#### ③ 血液常規與肝指數")
        hgb = st.slider("血紅素 (LBXHGB, g/dL)", 8.0, 19.0, float(dv["LBXHGB"]), 0.1)
        rbc = st.slider("紅血球數 (LBXRBCSI, 10⁶/μL)", 2.5, 7.5, float(dv["LBXRBCSI"]), 0.05)
        hct = st.slider("血球容積比 (LBXHCT, %)", 24.0, 55.0, float(dv["LBXHCT"]), 0.5)
        ast = st.slider("AST 肝指數 (LBXSASSI, U/L)", 6.0, 120.0, float( min(dv["LBXSASSI"], 120.0) ), 1.0)
        ggt = st.slider("GGT 肝指數 (LBXSGTSI, U/L)", 4.0, 200.0, float( min(dv["LBXSGTSI"], 200.0) ), 1.0)

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

    # 執行真實聯邦全域模型推論 (跨 Seeds 10~14)
    x_scaled = build_feature_vector(current_raw, scaler)
    seed_probs = []
    for s in [10, 11, 12, 13, 14]:
        npz_p = f"results/final/E2_lr0.01/seed{s}/global_model.npz"
        if os.path.exists(npz_p):
            seed_probs.append({"Seed": f"Seed {s}", "Prob": predict_with_npz(x_scaled, npz_p) * 100.0})

    mean_prob = float(np.mean([r["Prob"] for r in seed_probs]))
    std_prob = float(np.std([r["Prob"] for r in seed_probs], ddof=0))

    # 計算單一特徵邊際貢獻（將單一特徵替換為健康中位數後之預測差值）
    contrib_rows = []
    npz_ref = "results/final/E2_lr0.01/seed10/global_model.npz"
    base_p = predict_with_npz(x_scaled, npz_ref) * 100.0
    for feat in RAW_FEATURES:
        alt_raw = dict(current_raw)
        alt_raw[feat] = float(healthy_medians[feat])
        alt_p = predict_with_npz(build_feature_vector(alt_raw, scaler), npz_ref) * 100.0
        contrib_rows.append({
            "特徵": FEATURE_LABELS.get(feat, feat),
            "風險影響 (百分點)": base_p - alt_p,
        })
    contrib_df = pd.DataFrame(contrib_rows).sort_values("風險影響 (百分點)", key=lambda s: s.abs(), ascending=True).tail(8)

    res_left, res_mid, res_right = st.columns([1.3, 1.1, 1.6])

    with res_left:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_prob,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": f"DiabetesNet 預測風險機率 (±{std_prob:.2f}%)", "font": {"size": 16}},
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
        seed_df = pd.DataFrame(seed_probs)
        fig_seeds = px.bar(
            seed_df,
            x="Seed",
            y="Prob",
            text=seed_df["Prob"].map(lambda v: f"{v:.1f}%"),
            title="5 組最終全域模型 (Seeds 10~14) 預測一致性",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_seeds.add_hline(y=50, line_dash="dash", line_color="#ef4444")
        fig_seeds.update_yaxes(range=[0, 105], title="預測風險機率 (%)")
        fig_seeds.update_layout(height=360, xaxis_title="")
        st.plotly_chart(fig_seeds, use_container_width=True)

    with res_right:
        contrib_df["方向"] = np.where(contrib_df["風險影響 (百分點)"] >= 0, "推升風險", "降低風險")
        fig_contrib = px.bar(
            contrib_df,
            x="風險影響 (百分點)",
            y="特徵",
            orientation="h",
            color="方向",
            text=contrib_df["風險影響 (百分點)"].map(lambda v: f"{v:+.1f} pp"),
            title="真實模型個別特徵影響拆解 (相對於健康中位數)",
            color_discrete_map={"推升風險": "#ef4444", "降低風險": "#10b981"},
        )
        fig_contrib.update_layout(height=360, xaxis_title="對預測機率之邊際影響 (百分點 pp)", yaxis_title="", showlegend=False)
        st.plotly_chart(fig_contrib, use_container_width=True)
