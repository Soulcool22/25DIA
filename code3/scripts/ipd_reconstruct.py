import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter
except Exception:
    KaplanMeierFitter = None

BASE_DIR = Path("f:/25DIA/code3")
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class KMPoints:
    time: np.ndarray
    surv: np.ndarray
    at_risk: Optional[np.ndarray] = None


def load_km_points(csv_path: Path) -> KMPoints:
    df = pd.read_csv(csv_path)
    cols = [c.strip().lower() for c in df.columns]
    # Accept multiple header variants
    def get_col(name_options: List[str]):
        for n in name_options:
            if n in [c.strip().lower() for c in df.columns]:
                return n
        raise ValueError(f"列缺失: 可接受列名: {name_options}，在 {csv_path}")

    t_col = get_col(["time", "t", "time_months"]) 
    s_col = get_col(["s", "survival", "surv"]) 
    ar_col = None
    for opt in ["at_risk", "atrisk", "n_at_risk", "risk"]:
        if opt in cols:
            ar_col = opt
            break
    time = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    surv = pd.to_numeric(df[s_col], errors="coerce").to_numpy(dtype=float)
    at_risk = pd.to_numeric(df[ar_col], errors="coerce").to_numpy(dtype=float) if ar_col else None
    # sort by time
    idx = np.argsort(time)
    return KMPoints(time=time[idx], surv=surv[idx], at_risk=None if at_risk is None else at_risk[idx])


def reconstruct_ipd_piecewise_exp(km: KMPoints, total_n: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    简化的伪IPD重构：假设区间内常数风险率，使用S(t)分段近似，结合(可选)at-risk近似分配事件与删失。
    注意：该方法为近似，建议尽量提供区间人数at risk以提升准确度。
    """
    rng = np.random.default_rng(seed)
    t = km.time
    s = km.surv
    if total_n is None:
        # 若无at risk信息，假设初始样本量为100
        total_n = int(km.at_risk[0]) if km.at_risk is not None else 100
    # 初始化存活个体的“潜在”生存时间
    times = np.full(total_n, np.inf, dtype=float)
    events = np.zeros(total_n, dtype=int)

    # 计算每段的生存比与区间风险率
    for k in range(len(t) - 1):
        t0, t1 = t[k], t[k + 1]
        s0, s1 = s[k], s[k + 1]
        dt = max(t1 - t0, 1e-8)
        # 若生存概率上升或相等，认为无事件
        if s1 >= s0:
            continue
        # piecewise constant hazard implied by S1/S0
        hazard = -np.log(max(s1, 1e-12) / max(s0, 1e-12)) / dt
        # 当前仍在风险中的个体索引
        at_risk_idx = np.where((times == np.inf) & (events == 0))[0]
        n_risk = len(at_risk_idx)
        if n_risk == 0:
            break
        # 预期事件数（泊松近似）
        p_event_interval = 1 - np.exp(-hazard * dt)
        exp_events = int(round(n_risk * p_event_interval))
        exp_events = min(exp_events, n_risk)
        if exp_events <= 0:
            continue
        # 随机选择发生事件的个体，并为其生成事件时间（截断在区间内的指数分布）
        chosen = rng.choice(at_risk_idx, size=exp_events, replace=False)
        # 生成指数分布事件时间并截断在(t0, t1]
        u = rng.uniform(size=exp_events)
        event_times = t0 - np.log(1 - u * (1 - np.exp(-hazard * dt))) / hazard
        event_times = np.clip(event_times, t0 + 1e-6, t1)
        times[chosen] = event_times
        events[chosen] = 1
        # 其余个体留待下个区间，删失在末尾统一处理

    # 对剩余未发生事件的个体进行末次删失
    cens_idx = np.where((times == np.inf) & (events == 0))[0]
    if cens_idx.size > 0:
        last_time = t[-1]
        times[cens_idx] = last_time
        events[cens_idx] = 0

    df_ipd = pd.DataFrame({
        "time": times,
        "event": events
    }).sort_values("time").reset_index(drop=True)
    return df_ipd


def km_plot(df_ipd: pd.DataFrame, label: str, out_svg: Path):
    if KaplanMeierFitter is None:
        raise ImportError("需要lifelines库绘制KM曲线。")
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    kmf.fit(df_ipd["time"], event_observed=df_ipd["event"], label=label)
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("时间 (月)")
    ax.set_ylabel("生存概率")
    ax.set_title(f"重构KM曲线 - {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_svg = FIG_DIR / out_svg
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    return out_svg


def main():
    import argparse
    parser = argparse.ArgumentParser(description="从数字化KM点重构伪IPD")
    parser.add_argument("--points_csv", type=str, required=True, help="包含time与survival列的CSV路径")
    parser.add_argument("--label", type=str, required=True, help="曲线标签")
    parser.add_argument("--total_n", type=int, default=None, help="总样本量(可选,若提供at_risk则忽略)")
    parser.add_argument("--out_ipd", type=str, required=True, help="输出伪IPD CSV相对路径, 相对于results/")
    args = parser.parse_args()

    km = load_km_points(Path(args.points_csv))
    ipd = reconstruct_ipd_piecewise_exp(km, total_n=args.total_n)
    out_ipd_path = RES_DIR / args.out_ipd
    out_ipd_path.parent.mkdir(parents=True, exist_ok=True)
    ipd.to_csv(out_ipd_path, index=False)
    print(f"已保存伪IPD: {out_ipd_path}")
    svg_path = km_plot(ipd, args.label, Path(Path(args.out_ipd).with_suffix('.svg').name))
    print(f"已保存重构KM图: {svg_path}")

if __name__ == "__main__":
    main()