from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

BASE_DIR = Path("f:/25DIA/code3.1")
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


def overlay_two_km(ipd1_csv: Path, label1: str, ipd2_csv: Path, label2: str, out_svg: Path):
    df1 = pd.read_csv(ipd1_csv)
    df2 = pd.read_csv(ipd2_csv)
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    kmf1.fit(df1["time"], event_observed=df1["event"], label=label1)
    kmf1.plot_survival_function(ax=ax, ci_show=False)
    kmf2.fit(df2["time"], event_observed=df2["event"], label=label2)
    kmf2.plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlabel("时间 (月)")
    ax.set_ylabel("生存概率")
    ax.set_title("两研究KM叠加图")
    ax.grid(True, alpha=0.3)
    out_svg = FIG_DIR / out_svg
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    return out_svg


def main():
    import argparse
    parser = argparse.ArgumentParser(description="叠加两个伪IPD的KM曲线")
    parser.add_argument("--ipd1_csv", type=str, required=True)
    parser.add_argument("--label1", type=str, required=True)
    parser.add_argument("--ipd2_csv", type=str, required=True)
    parser.add_argument("--label2", type=str, required=True)
    parser.add_argument("--out_svg", type=str, required=True)
    args = parser.parse_args()

    out = overlay_two_km(Path(args.ipd1_csv), args.label1, Path(args.ipd2_csv), args.label2, Path(args.out_svg))
    print(f"已保存KM叠加图: {out}")

if __name__ == "__main__":
    main()