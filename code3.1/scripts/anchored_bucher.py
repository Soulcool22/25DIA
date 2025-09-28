from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path("f:/25DIA/code3.1")
RES_DIR = BASE_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def read_hr_from_cox_summary(csv_path: Path, var_name: str | None = None):
    df = pd.read_csv(csv_path, index_col=0)
    if var_name is not None and var_name in df.index:
        row = df.loc[var_name]
    else:
        # 自动探测治疗指示变量行
        candidates = [idx for idx in df.index if any(k in str(idx).lower() for k in ["trt", "treat", "treatment", "group", "arm"]) and "sex" not in str(idx).lower() and "ecog" not in str(idx).lower() and "age" not in str(idx).lower()]
        if not candidates:
            raise ValueError(f"未能在 {csv_path} 中识别治疗变量行，请提供 var_name 参数或检查CSV索引。可用索引: {list(df.index)}")
        row = df.loc[candidates[0]]
    return float(row["exp(coef)"]), float(row["exp(coef) lower 95%"]), float(row["exp(coef) upper 95%"])


def bucher_ratio(hr_A_vs_O: float, hr_B_vs_O: float) -> float:
    # 锚定Bucher: HR_A_vs_B = HR_A_vs_O / HR_B_vs_O
    return hr_A_vs_O / hr_B_vs_O


def combine_ci_log(hr: float, ci_low: float, ci_up: float):
    # 使用log(HR)近似的标准误以合并不确定度
    se = (np.log(ci_up) - np.log(ci_low)) / (2 * 1.96)
    return np.log(hr), se


def bucher_ci(hr_A_vs_O, ci_A, hr_B_vs_O, ci_B):
    logA, seA = combine_ci_log(hr_A_vs_O, *ci_A)
    logB, seB = combine_ci_log(hr_B_vs_O, *ci_B)
    logAB = logA - logB
    seAB = np.sqrt(seA**2 + seB**2)
    hrAB = np.exp(logAB)
    ci_low = np.exp(logAB - 1.96 * seAB)
    ci_up = np.exp(logAB + 1.96 * seAB)
    return hrAB, ci_low, ci_up


def main():
    import argparse
    parser = argparse.ArgumentParser(description="锚定Bucher法计算A vs B的HR")
    parser.add_argument("--cox_A_vs_O", type=str, required=True, help="RCT-A加权或未加权Cox结果CSV路径")
    parser.add_argument("--cox_B_vs_O", type=str, required=True, help="RCT-B/C重构IPD或摘要Cox结果CSV路径")
    parser.add_argument("--out_csv", type=str, required=True, help="输出Bucher结果相对路径(相对results/)")
    parser.add_argument("--var_A", type=str, default=None, help="A vs O Cox CSV中治疗变量索引名(可选)")
    parser.add_argument("--var_B", type=str, default=None, help="B/C vs O Cox CSV中治疗变量索引名(可选)")
    args = parser.parse_args()

    hrA, lA, uA = read_hr_from_cox_summary(Path(args.cox_A_vs_O), var_name=args.var_A)
    hrB, lB, uB = read_hr_from_cox_summary(Path(args.cox_B_vs_O), var_name=args.var_B)
    hrAB, ciL, ciU = bucher_ci(hrA, (lA, uA), hrB, (lB, uB))

    out_path = RES_DIR / args.out_csv
    pd.DataFrame({"HR": [hrAB], "CI_low": [ciL], "CI_up": [ciU]}).to_csv(out_path, index=False)
    print(f"已保存Bucher结果: {out_path}")

if __name__ == "__main__":
    main()