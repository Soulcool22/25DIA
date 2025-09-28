from pathlib import Path
import math
import numpy as np
import pandas as pd

BASE_DIR = Path("f:/25DIA/code3")
RES_DIR = BASE_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def read_hr_from_cox_summary(path: Path, term: str = "trt_drug_a"):
    df = pd.read_csv(path, index_col=0)
    row = df.loc[term]
    logHR = float(row["coef"])  # lifelines stores log(HR)
    se = float(row["se(coef)"])
    HR = math.exp(logHR)
    lcl = float(row["exp(coef) lower 95%"])
    ucl = float(row["exp(coef) upper 95%"])
    return {"HR": HR, "logHR": logHR, "se": se, "lcl": lcl, "ucl": ucl}


def combine_bucher(hr_a_o: dict, hr_b_o: dict):
    # Anchored: log(HR_A/B) = log(HR_A/O) - log(HR_B/O)
    logHR = hr_a_o["logHR"] - math.log(hr_b_o["HR"]) if "logHR" not in hr_b_o else hr_a_o["logHR"] - hr_b_o["logHR"]
    # SE: assume independence
    se = math.sqrt(hr_a_o.get("se", 0.0) ** 2 + hr_b_o.get("se", 0.0) ** 2)
    HR = math.exp(logHR)
    lcl = math.exp(logHR - 1.96 * se) if se > 0 else np.nan
    ucl = math.exp(logHR + 1.96 * se) if se > 0 else np.nan
    return {"HR": HR, "lcl": lcl, "ucl": ucl, "logHR": logHR, "se": se}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="锚定间接比较（Bucher法）计算 A 对 B/C 的相对疗效")
    parser.add_argument("--a_o_csv", required=True, help="A对O的Cox结果CSV路径（来自RCT-A或加权后的MAIC）")
    parser.add_argument("--b_or_c_o_csv", required=True, help="B或C对O的Cox结果CSV路径（来自重构IPD或报告）")
    parser.add_argument("--term_a", default="trt_drug_a", help="A/O文件中治疗项名称，默认trt_drug_a")
    parser.add_argument("--term_b", default="trt_drug_a", help="B或C/O文件中治疗项名称，若药物作为指示变量同名可不改")
    parser.add_argument("--out", required=True, help="输出文件名（results目录下）")
    args = parser.parse_args()

    a_o = read_hr_from_cox_summary(Path(args.a_o_csv), term=args.term_a)
    b_o = read_hr_from_cox_summary(Path(args.b_or_c_o_csv), term=args.term_b)
    res = combine_bucher(a_o, b_o)

    out_path = RES_DIR / args.out
    pd.DataFrame([res]).to_csv(out_path, index=False)
    print(f"已保存锚定比较结果: {out_path}")
    print(res)

if __name__ == "__main__":
    main()