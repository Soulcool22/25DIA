import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import root
from lifelines import CoxPHFitter

BASE_DIR = Path("f:/25DIA/code3")
DATA_PATH = Path("f:/25DIA/复赛大题（三）数据集.csv")
RES_DIR = BASE_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class TargetSpec:
    name: str
    target: float


def load_targets(csv_path: Path) -> List[TargetSpec]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"variable", "target_value"}
    if not required.issubset(df.columns):
        raise ValueError(f"目标配置文件需包含列: {required}, 实际列: {df.columns.tolist()}")
    specs = [TargetSpec(row["variable"], float(row["target_value"])) for _, row in df.iterrows()]
    return specs


def build_covariate_matrix(df: pd.DataFrame, specs: List[TargetSpec]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Derive helper columns consistent with other scripts
    df = df.copy()
    df["sex_male"] = (df["sex"].astype(str) == "Male").astype(int)
    df["ecog_2"] = (df["ecog"].astype(str) == "2").astype(int)
    X_cols = []
    target_vec = []
    for s in specs:
        if s.name not in df.columns:
            raise ValueError(f"数据中缺少指定变量: {s.name}")
        X_cols.append(s.name)
        target_vec.append(s.target)
    X = df[X_cols].to_numpy(dtype=float)
    target = np.asarray(target_vec, dtype=float)
    return X, target, X_cols


def solve_maic_weights(X: np.ndarray, target: np.ndarray) -> np.ndarray:
    # w_i ∝ exp(alpha^T x_i), solve for alpha s.t. weighted mean equals target
    n, p = X.shape
    def moment_cond(alpha: np.ndarray) -> np.ndarray:
        w = np.exp(X @ alpha)
        w /= w.mean()  # normalize mean weight = 1
        mu = (w[:, None] * X).sum(axis=0) / w.sum()
        return mu - target
    sol = root(moment_cond, x0=np.zeros(X.shape[1]))
    if not sol.success:
        raise RuntimeError(f"MAIC求解失败: {sol.message}")
    alpha = sol.x
    w = np.exp(X @ alpha)
    w /= w.mean()  # mean weight = 1 for stability
    return w


def fit_weighted_cox(df: pd.DataFrame, weights: np.ndarray) -> CoxPHFitter:
    df = df.copy()
    df["trt_drug_a"] = (df["treatment"].astype(str) == "drug_a").astype(int)
    df["weight"] = weights
    cols = ["age", "trt_drug_a", "sex_male", "ecog_2", "weight", "time(OS)", "event"]
    cph = CoxPHFitter()
    cph.fit(df[cols], duration_col="time(OS)", event_col="event", weights_col="weight", robust=True)
    return cph


def main():
    import argparse
    parser = argparse.ArgumentParser(description="对RCT-A IPD进行MAIC加权以匹配目标人群基线")
    parser.add_argument("--biomarker", required=True, choices=["Negative", "Positive"], help="生物标志物分层")
    parser.add_argument("--targets_csv", required=True, help="目标人群基线配置CSV，列: variable,target_value")
    parser.add_argument("--out_prefix", required=True, help="输出前缀名称，用于结果文件命名")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    # 统一事件列名
    event_col_candidates = [c for c in df.columns if c.lower().startswith("event")] 
    df = df.rename(columns={event_col_candidates[0]: "event"})
    # 仅RCT-A
    df = df[df["study"] == "RCT_A_vs_O"].copy()
    # 类型与派生
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["time(OS)"] = pd.to_numeric(df["time(OS)"] , errors="coerce")
    df["sex"] = df["sex"].astype("category")
    df["ecog"] = df["ecog"].astype("category")
    df["biomarker_x"] = df["biomarker_x"].astype("category")
    df = df[df["biomarker_x"] == args.biomarker].copy()

    specs = load_targets(Path(args.targets_csv))
    X, target, names = build_covariate_matrix(df, specs)
    w = solve_maic_weights(X, target)

    # 保存权重
    w_out = df[["patient_id"]].copy()
    w_out["weight"] = w
    w_path = RES_DIR / f"weights_{args.out_prefix}.csv"
    w_out.to_csv(w_path, index=False)

    # 加权Cox
    cph = fit_weighted_cox(df, w)
    cph.summary.to_csv(RES_DIR / f"cox_maic_{args.out_prefix}.csv")

    # 打印关键结果
    tr = cph.summary.loc["trt_drug_a"]
    print({
        "HR": float(np.exp(tr["coef"])),
        "HR_CI_lower": float(tr["exp(coef) lower 95%"]),
        "HR_CI_upper": float(tr["exp(coef) upper 95%"]),
        "se_logHR": float(tr["se(coef)"])
    })
    print(f"已保存权重: {w_path}")

if __name__ == "__main__":
    main()