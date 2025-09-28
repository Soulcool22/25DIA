from pathlib import Path
import subprocess
import sys

BASE_DIR = Path("f:/25DIA/code3.1")
DATA_DIR = BASE_DIR / "data"
RES_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
SCRIPTS = BASE_DIR / "scripts"


def run(cmd):
    print("运行:", " ".join(cmd))
    res = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(res.returncode)


def main():
    # 1) 分析RCT-A (输出KM与Cox)
    run([str(SCRIPTS / "rct_a_analysis.py")])

    # 2) 重构RCT-B (X阴性) 与 RCT-C (X阳性) 的伪IPD
    # 需用户将数字化点保存为 data/rct_b_xneg_km.csv 和 data/rct_c_xpos_km.csv
    b_points = DATA_DIR / "rct_b_xneg_km.csv"
    c_points = DATA_DIR / "rct_c_xpos_km.csv"
    if b_points.exists():
        run([str(SCRIPTS / "ipd_reconstruct.py"), "--points_csv", str(b_points), "--label", "RCT-B X阴性", "--total_n", "200", "--out_ipd", "rct_b_xneg_ipd.csv"])  # total_n可调整
    else:
        print("缺少: data/rct_b_xneg_km.csv，跳过RCT-B重构")
    if c_points.exists():
        run([str(SCRIPTS / "ipd_reconstruct.py"), "--points_csv", str(c_points), "--label", "RCT-C X阳性", "--total_n", "200", "--out_ipd", "rct_c_xpos_ipd.csv"])  # total_n可调整
    else:
        print("缺少: data/rct_c_xpos_km.csv，跳过RCT-C重构")

    # 3) 叠加KM (A vs B 在X阴性)
    a_neg_ipd = RES_DIR / "rct_a_xneg_ipd.csv"
    b_neg_ipd = RES_DIR / "rct_b_xneg_ipd.csv"
    if a_neg_ipd.exists() and b_neg_ipd.exists():
        run([str(SCRIPTS / "overlay_km.py"), "--ipd1_csv", str(a_neg_ipd), "--label1", "RCT-A X阴性", "--ipd2_csv", str(b_neg_ipd), "--label2", "RCT-B X阴性", "--out_svg", "overlay_a_b_xneg.svg"]) 

if __name__ == "__main__":
    main()