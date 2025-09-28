import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
except Exception as e:
    raise ImportError("缺少lifelines库，请在运行环境安装：pip install lifelines")

BASE_DIR = Path("f:/25DIA/code3.1")
DATA_PATH = Path("f:/25DIA/复赛大题（三）数据集.csv")
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

CHINESE_FONT_FAMILIES = [
    "Microsoft YaHei", "SimHei", "Microsoft JhengHei", "KaiTi", "SimSun"
]

def setup_cn_fonts():
    for f in CHINESE_FONT_FAMILIES:
        plt.rcParams["font.sans-serif"] = [f]
        plt.rcParams["axes.unicode_minus"] = False
        break


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据集不存在: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    event_col_candidates = [c for c in df.columns if c.lower().startswith("event")]
    if not event_col_candidates:
        raise ValueError("未找到事件指示列(event)。")
    event_col = event_col_candidates[0]
    df = df.rename(columns={event_col: "event"})
    df["time(OS)"] = pd.to_numeric(df["time(OS)"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = df["sex"].astype("category")
    df["ecog"] = df["ecog"].astype("category")
    df["biomarker_x"] = df["biomarker_x"].astype("category")
    df["treatment"] = df["treatment"].astype("category")
    df = df[df["study"] == "RCT_A_vs_O"].copy()
    return df


def km_plot_by_group(df: pd.DataFrame, biomarker_level: str, filename: Path):
    setup_cn_fonts()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    kmf = KaplanMeierFitter()
    subset = df[df["biomarker_x"] == biomarker_level]
    for trt, sub in subset.groupby("treatment"):
        kmf.fit(durations=sub["time(OS)"], event_observed=sub["event"], label=f"{trt}")
        kmf.plot_survival_function(ax=ax)
    ax.set_title(f"RCT-A OS KM曲线 - 生物标志物X {biomarker_level}")
    ax.set_xlabel("时间 (月)")
    ax.set_ylabel("生存概率")
    ax.grid(True, alpha=0.3)
    filename = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(filename, format="svg")
    plt.close(fig)
    return filename


def fit_cox_and_save(df: pd.DataFrame, biomarker_level: str, filename: Path):
    sub = df[df["biomarker_x"] == biomarker_level].copy()
    sub["trt_drug_a"] = (sub["treatment"] == "drug_a").astype(int)
    sub["sex_male"] = (sub["sex"] == "Male").astype(int)
    sub["ecog_2"] = (sub["ecog"].astype(str) == "2").astype(int)

    cols = ["age", "trt_drug_a", "sex_male", "ecog_2"]
    cph = CoxPHFitter()
    cph.fit(sub[cols + ["time(OS)", "event"]], duration_col="time(OS)", event_col="event")
    out_path = RES_DIR / filename
    cph.summary.to_csv(out_path, index=True)
    return out_path, cph


def main():
    df = load_data()
    neg_fig = km_plot_by_group(df, "Negative", Path("rct_a_km_x_negative.svg"))
    pos_fig = km_plot_by_group(df, "Positive", Path("rct_a_km_x_positive.svg"))

    # 额外输出：按生物标志物分层的伪IPD（直接使用真实IPD的time/event两列）
    df_neg = df[df["biomarker_x"] == "Negative"][["time(OS)", "event"]].rename(columns={"time(OS)": "time"})
    df_pos = df[df["biomarker_x"] == "Positive"][["time(OS)", "event"]].rename(columns={"time(OS)": "time"})
    (RES_DIR / "rct_a_xneg_ipd.csv").parent.mkdir(parents=True, exist_ok=True)
    df_neg.to_csv(RES_DIR / "rct_a_xneg_ipd.csv", index=False)
    df_pos.to_csv(RES_DIR / "rct_a_xpos_ipd.csv", index=False)

    neg_res_path, neg_cph = fit_cox_and_save(df, "Negative", Path("cox_x_negative.csv"))
    pos_res_path, pos_cph = fit_cox_and_save(df, "Positive", Path("cox_x_positive.csv"))

    def extract_hr(cph_obj):
        row = cph_obj.summary.loc["trt_drug_a"]
        return {
            "HR": float(np.exp(row["coef"])),
            "HR_CI_lower": float(row["exp(coef) lower 95%"]),
            "HR_CI_upper": float(row["exp(coef) upper 95%"]),
            "p": float(row["p"]),
        }
    print("RCT-A治疗效应 (X阴性):", extract_hr(neg_cph))
    print("RCT-A治疗效应 (X阳性):", extract_hr(pos_cph))

if __name__ == "__main__":
    main()