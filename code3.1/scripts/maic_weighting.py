from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

BASE_DIR = Path("f:/25DIA/code3.1")
RES_DIR = BASE_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def load_target_specs(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    # 需包含列: covariate, mean (或 proportion)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"covariate", "mean"}.issubset(df.columns):
        raise ValueError("目标人群特征CSV需包含: covariate, mean 两列")
    return pd.Series(df["mean"].values, index=df["covariate"].values)


def build_covariate_matrix(df: pd.DataFrame, covariates: list) -> np.ndarray:
    X = []
    for c in covariates:
        if c not in df.columns:
            raise ValueError(f"协变量 {c} 不存在于RCT-A IPD数据中")
        X.append(pd.to_numeric(df[c], errors="coerce").values.reshape(-1, 1))
    return np.hstack(X)


def solve_maic_weights(X: np.ndarray, target_means: np.ndarray, max_iter: int = 500, lr: float = 0.1):
    # 优化theta使得加权样本均值等于目标均值: w_i = exp(X_i^T theta)
    n, p = X.shape
    theta = np.zeros(p)
    for _ in range(max_iter):
        w = np.exp(X @ theta)
        w = w / w.mean()
        grad = X.T @ (w / w.sum()) - target_means
        theta -= lr * grad
        if np.linalg.norm(grad) < 1e-6:
            break
    w = np.exp(X @ theta)
    w = w / w.mean()
    return w


def fit_weighted_cox(df: pd.DataFrame, weight_col: str, duration_col: str = "time(OS)", event_col: str = "event"):
    cols = [duration_col, event_col, "treatment", weight_col]
    tmp = df[cols].copy()
    tmp["trt_drug_a"] = (df["treatment"] == "drug_a").astype(int)
    cph = CoxPHFitter()
    cph.fit(tmp[[duration_col, event_col, "trt_drug_a", weight_col]], duration_col=duration_col, event_col=event_col, weights_col=weight_col)
    return cph


def main():
    import argparse
    parser = argparse.ArgumentParser(description="对RCT-A IPD进行MAIC加权以匹配目标人群特征")
    parser.add_argument("--rct_a_csv", type=str, required=True, help="RCT-A IPD CSV路径 (包含time(OS), event, treatment等列)")
    parser.add_argument("--biomarker_level", type=str, required=True, choices=["Negative", "Positive"], help="X阴性或阳性子集")
    parser.add_argument("--target_specs_csv", type=str, required=True, help="目标人群特征CSV，含covariate, mean列")
    parser.add_argument("--out_weights_csv", type=str, required=True, help="输出权重文件(相对于results/)")
    args = parser.parse_args()

    df = pd.read_csv(args.rct_a_csv)
    df = df[df["biomarker_x"] == args.biomarker_level].copy()
    specs = load_target_specs(Path(args.target_specs_csv))
    covariates = list(specs.index)
    target_means = specs.values

    X = build_covariate_matrix(df, covariates)
    w = solve_maic_weights(X, target_means)
    df["maic_w"] = w

    out_path = RES_DIR / args.out_weights_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["maic_w"]].to_csv(out_path, index=False)
    print(f"已保存权重: {out_path}")

    # 可选: 拟合加权Cox以得到对比Drug-A vs O的HR
    try:
        cph = fit_weighted_cox(df, "maic_w")
        print(cph.summary)
    except Exception as e:
        print(f"加权Cox拟合失败: {e}")

if __name__ == "__main__":
    main()