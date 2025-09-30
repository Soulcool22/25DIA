import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
BASE_DIR = Path("f:/25DIA/code3.2")
DATA_PATH = Path("f:/25DIA/复赛大题（三）数据集.csv")
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
class IPDReconstructor:
    def __init__(self):
        self.rct_a_data = None
        self.rct_b_published_data = None
    def load_rct_a_data(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"数据集不存在: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        event_col_candidates = [c for c in df.columns if 'event' in c.lower()]
        if event_col_candidates:
            df = df.rename(columns={event_col_candidates[0]: "event"})
        df["time(OS)"] = pd.to_numeric(df["time(OS)"], errors="coerce")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df = df[df["study"] == "RCT_A_vs_O"].copy()
        self.rct_a_data = df
        return df
    def setup_rct_b_published_data(self):
        self.rct_b_published_data = {
            'itt': {
                'drug_b': {'n': 119, 'age_le_65_pct': 68.1, 'male_pct': 54.6},
                'control': {'n': 121, 'age_le_65_pct': 66.1, 'male_pct': 61.2}
            },
            'x_positive': {
                'drug_b': {'n': 49, 'age_le_65_pct': 67.3, 'male_pct': 80.0},
                'control': {'n': 55, 'age_le_65_pct': 80.0, 'male_pct': 80.0}
            }
        }
        self.rct_b_published_data['x_negative'] = {
            'drug_b': {'n': 119 - 49},
            'control': {'n': 121 - 55}
        }
        for arm in ['drug_b', 'control']:
            itt_data = self.rct_b_published_data['itt'][arm]
            x_pos_data = self.rct_b_published_data['x_positive'][arm]
            x_neg_data = self.rct_b_published_data['x_negative'][arm]
            n_itt = itt_data['n']
            n_pos = x_pos_data['n']
            n_neg = x_neg_data['n']
            neg_age_le_65_pct = (itt_data['age_le_65_pct'] * n_itt - 
                                x_pos_data['age_le_65_pct'] * n_pos) / n_neg
            neg_male_pct = (itt_data['male_pct'] * n_itt - 
                           x_pos_data['male_pct'] * n_pos) / n_neg
            x_neg_data['age_le_65_pct'] = max(0, min(100, neg_age_le_65_pct))
            x_neg_data['male_pct'] = max(0, min(100, neg_male_pct))
    def estimate_survival_parameters(self, subgroup='x_negative'):
        if self.rct_a_data is None:
            self.load_rct_a_data()
        if subgroup == 'x_negative':
            subset = self.rct_a_data[self.rct_a_data['biomarker_x'] == 'Negative']
        else:
            subset = self.rct_a_data[self.rct_a_data['biomarker_x'] == 'Positive']
        survival_params = {}
        for trt in ['drug_a', 'control']:
            trt_data = subset[subset['treatment'] == trt]
            kmf = KaplanMeierFitter()
            kmf.fit(trt_data['time(OS)'], trt_data['event'])
            median_surv = kmf.median_survival_time_
            times = trt_data['time(OS)'].values
            events = trt_data['event'].values
            total_time = np.sum(times)
            total_events = np.sum(events)
            hazard_rate = total_events / total_time if total_time > 0 else 0.1
            survival_params[trt] = {
                'hazard_rate': hazard_rate,
                'median_survival': median_surv,
                'n_patients': len(trt_data),
                'n_events': total_events
            }
        return survival_params
    def reconstruct_rct_b_x_negative(self):
        surv_params = self.estimate_survival_parameters('x_negative')
        n_drug_b = self.rct_b_published_data['x_negative']['drug_b']['n']
        n_control = self.rct_b_published_data['x_negative']['control']['n']
        reconstructed_data = []
        np.random.seed(42)
        drug_b_hr = 0.75
        for arm, n_patients in [('drug_b', n_drug_b), ('control', n_control)]:
            base_hazard = surv_params['control']['hazard_rate']
            if arm == 'drug_b':
                hazard_rate = base_hazard * drug_b_hr
            else:
                hazard_rate = base_hazard
            survival_times = np.random.exponential(1/hazard_rate, n_patients)
            censoring_times = np.random.uniform(12, 24, n_patients)
            observed_times = np.minimum(survival_times, censoring_times)
            events = (survival_times <= censoring_times).astype(int)
            baseline_data = self.rct_b_published_data['x_negative'][arm]
            age_le_65_prob = baseline_data['age_le_65_pct'] / 100
            ages = np.where(np.random.random(n_patients) < age_le_65_prob,
                           np.random.normal(60, 8, n_patients),
                           np.random.normal(70, 6, n_patients))
            ages = np.clip(ages, 18, 85)
            male_prob = baseline_data['male_pct'] / 100
            sexes = np.where(np.random.random(n_patients) < male_prob, 'Male', 'Female')
            ecog_scores = np.random.choice(['0-1', '2'], n_patients, p=[0.7, 0.3])
            for i in range(n_patients):
                reconstructed_data.append({
                    'patient_id': f'RCTB_{arm}_{i+1:03d}',
                    'study': 'RCT_B_vs_O',
                    'treatment': arm if arm != 'drug_b' else 'drug_b',
                    'age': ages[i],
                    'sex': sexes[i],
                    'ecog': ecog_scores[i],
                    'biomarker_x': 'Negative',
                    'time(OS)': observed_times[i],
                    'event': events[i]
                })
        return pd.DataFrame(reconstructed_data)
    def create_overlay_km_plot(self, rct_b_reconstructed):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
        rct_a_x_neg = self.rct_a_data[self.rct_a_data['biomarker_x'] == 'Negative']
        kmf = KaplanMeierFitter()
        for trt in ['drug_a', 'control']:
            subset = rct_a_x_neg[rct_a_x_neg['treatment'] == trt]
            label = f'RCT-A {trt.replace("_", "-").title()}'
            kmf.fit(subset['time(OS)'], subset['event'], label=label)
            kmf.plot_survival_function(ax=ax, linestyle='-', linewidth=2)
        for trt in ['drug_b', 'control']:
            subset = rct_b_reconstructed[rct_b_reconstructed['treatment'] == trt]
            label = f'RCT-B {trt.replace("_", "-").title()} (重构)'
            kmf.fit(subset['time(OS)'], subset['event'], label=label)
            kmf.plot_survival_function(ax=ax, linestyle='--', linewidth=2, alpha=0.8)
        ax.set_title('RCT-A vs 重构RCT-B X阴性亚组 OS Kaplan-Meier曲线对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间 (月)', fontsize=12)
        ax.set_ylabel('生存概率', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)            
        plt.tight_layout()
        output_path = FIG_DIR / "problem_3_1_overlay_km_curves.svg"
        fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    def analyze_reconstruction_limitations(self):
        return "重构完成"
def main():
    reconstructor = IPDReconstructor()
    reconstructor.load_rct_a_data()
    reconstructor.setup_rct_b_published_data()
    rct_b_reconstructed = reconstructor.reconstruct_rct_b_x_negative()
    output_data_path = RES_DIR / "rct_b_x_negative_reconstructed.csv"
    rct_b_reconstructed.to_csv(output_data_path, index=False)
    km_plot_path = reconstructor.create_overlay_km_plot(rct_b_reconstructed)
    limitations = reconstructor.analyze_reconstruction_limitations()
    overlay_plot_path = reconstructor.create_overlay_km_plot(rct_b_reconstructed)
    return {
        'reconstructed_data': rct_b_reconstructed,
        'overlay_plot_path': overlay_plot_path
    }
if __name__ == "__main__":
    results = main()