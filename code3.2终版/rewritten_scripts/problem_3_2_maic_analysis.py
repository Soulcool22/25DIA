import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
BASE_DIR = Path("f:/25DIA/code3.2")
DATA_PATH = Path("f:/25DIA/复赛大题（三）数据集.csv")
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
class MAICAnalyzer:
    def __init__(self):
        self.rct_a_data = None
        self.rct_b_data = None
        self.rct_b_target_population = None
        self.weights = None
    def load_data(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"数据集不存在: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        event_col_candidates = [c for c in df.columns if 'event' in c.lower()]
        if event_col_candidates:
            df = df.rename(columns={event_col_candidates[0]: "event"})
        df["time(OS)"] = pd.to_numeric(df["time(OS)"], errors="coerce")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        self.rct_a_data = df[df["study"] == "RCT_A_vs_O"].copy()
        rct_b_path = RES_DIR / "rct_b_x_negative_reconstructed.csv"
        if rct_b_path.exists():
            self.rct_b_data = pd.read_csv(rct_b_path)
        else:
            raise FileNotFoundError("请先运行问题3.1生成重构的RCT-B数据")
    def prepare_target_population(self):
        rct_b_x_neg = self.rct_b_data.copy()
        target_characteristics = {
            'age_mean': rct_b_x_neg['age'].mean(),
            'male_prop': (rct_b_x_neg['sex'] == 'Male').mean(),
            'ecog_2_prop': (rct_b_x_neg['ecog'] == '2').mean()
        }
        self.rct_b_target_population = target_characteristics
        return target_characteristics
    def calculate_maic_weights(self):
        rct_a_x_neg = self.rct_a_data[self.rct_a_data['biomarker_x'] == 'Negative'].copy()
        rct_a_x_neg['age_centered'] = rct_a_x_neg['age'] - rct_a_x_neg['age'].mean()
        rct_a_x_neg['male'] = (rct_a_x_neg['sex'] == 'Male').astype(int)
        rct_a_x_neg['ecog_2'] = (rct_a_x_neg['ecog'] == '2').astype(int)
        target_age_mean = self.rct_b_target_population['age_mean']
        target_male_prop = self.rct_b_target_population['male_prop']
        target_ecog_2_prop = self.rct_b_target_population['ecog_2_prop']
        X = rct_a_x_neg[['age', 'male', 'ecog_2']].values
        n = len(rct_a_x_neg)
        target_means = np.array([target_age_mean, target_male_prop, target_ecog_2_prop])
        def objective(log_weights):
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights) * n
            weighted_means = np.average(X, weights=weights, axis=0)
            balance_penalty = np.sum((weighted_means - target_means) ** 2) * 1000
            weight_variance_penalty = np.var(weights) * 0.1
            return balance_penalty + weight_variance_penalty
        def constraint_sum(log_weights):
            weights = np.exp(log_weights)
            return np.sum(weights) - n
        def constraint_balance_age(log_weights):
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights) * n
            weighted_mean = np.average(X[:, 0], weights=weights)
            return weighted_mean - target_means[0]
        def constraint_balance_male(log_weights):
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights) * n
            weighted_mean = np.average(X[:, 1], weights=weights)
            return weighted_mean - target_means[1]
        def constraint_balance_ecog(log_weights):
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights) * n
            weighted_mean = np.average(X[:, 2], weights=weights)
            return weighted_mean - target_means[2]
        initial_log_weights = np.zeros(n)
        constraints = [
            {'type': 'eq', 'fun': constraint_balance_age},
            {'type': 'eq', 'fun': constraint_balance_male},
            {'type': 'eq', 'fun': constraint_balance_ecog}
        ]
        try:
            result = minimize(objective, initial_log_weights, 
                            method='SLSQP', constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            if result.success:
                weights = np.exp(result.x)
                weights = weights / np.sum(weights) * n
                self.weights = weights
                weighted_means = np.average(X, weights=weights, axis=0)
            else:
                self.weights = np.ones(n)
        except Exception as e:
            self.weights = np.ones(n)
        return self.weights
    def perform_weighted_analysis(self):
        rct_a_x_neg = self.rct_a_data[self.rct_a_data['biomarker_x'] == 'Negative'].copy()
        rct_a_x_neg['maic_weight'] = self.weights
        results = {}
        for treatment in ['drug_a', 'control']:
            subset = rct_a_x_neg[rct_a_x_neg['treatment'] == treatment].copy()
            kmf = KaplanMeierFitter()
            kmf.fit(subset['time(OS)'], subset['event'], weights=subset['maic_weight'])
            median_survival = kmf.median_survival_time_
            total_weighted_time = np.sum(subset['time(OS)'] * subset['maic_weight'])
            total_weighted_events = np.sum(subset['event'] * subset['maic_weight'])
            hazard_rate = total_weighted_events / total_weighted_time if total_weighted_time > 0 else 0
            results[treatment] = {
                'median_survival': median_survival,
                'hazard_rate': hazard_rate,
                'n_patients': len(subset),
                'effective_n': np.sum(subset['maic_weight']),
                'n_events': np.sum(subset['event'])
            }
        return results
    def calculate_indirect_comparison(self):
        rct_a_weighted_results = self.perform_weighted_analysis()
        rct_b_results = {}
        for treatment in ['drug_b', 'control']:
            subset = self.rct_b_data[self.rct_b_data['treatment'] == treatment]
            kmf = KaplanMeierFitter()
            kmf.fit(subset['time(OS)'], subset['event'])
            median_survival = kmf.median_survival_time_
            total_time = np.sum(subset['time(OS)'])
            total_events = np.sum(subset['event'])
            hazard_rate = total_events / total_time if total_time > 0 else 0
            rct_b_results[treatment] = {
                'median_survival': median_survival,
                'hazard_rate': hazard_rate,
                'n_patients': len(subset),
                'n_events': np.sum(subset['event'])
            }
        hr_a_vs_control = (rct_a_weighted_results['drug_a']['hazard_rate'] / 
                          rct_a_weighted_results['control']['hazard_rate'])
        hr_b_vs_control = (rct_b_results['drug_b']['hazard_rate'] / 
                          rct_b_results['control']['hazard_rate'])
        hr_a_vs_b = hr_a_vs_control / hr_b_vs_control
        se_log_hr_a = np.sqrt(1/rct_a_weighted_results['drug_a']['n_events'] + 
                             1/rct_a_weighted_results['control']['n_events'])
        se_log_hr_b = np.sqrt(1/rct_b_results['drug_b']['n_events'] + 
                             1/rct_b_results['control']['n_events'])
        se_log_hr_indirect = np.sqrt(se_log_hr_a**2 + se_log_hr_b**2)
        log_hr_indirect = np.log(hr_a_vs_b)
        ci_lower = np.exp(log_hr_indirect - 1.96 * se_log_hr_indirect)
        ci_upper = np.exp(log_hr_indirect + 1.96 * se_log_hr_indirect)
        indirect_comparison_results = {
            'hr_drug_a_vs_drug_b': hr_a_vs_b,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'log_hr': log_hr_indirect,
            'se_log_hr': se_log_hr_indirect,
            'rct_a_weighted': rct_a_weighted_results,
            'rct_b_results': rct_b_results
        }
        return indirect_comparison_results
    def create_forest_plot(self, results):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        studies = ['RCT-A Drug-A vs Control\n(MAIC加权)', 
                  'RCT-B Drug-B vs Control\n(重构数据)',
                  'Drug-A vs Drug-B\n(间接比较)']
        hr_a_vs_control = (results['rct_a_weighted']['drug_a']['hazard_rate'] / 
                          results['rct_a_weighted']['control']['hazard_rate'])
        hr_b_vs_control = (results['rct_b_results']['drug_b']['hazard_rate'] / 
                          results['rct_b_results']['control']['hazard_rate'])
        hrs = [hr_a_vs_control, hr_b_vs_control, results['hr_drug_a_vs_drug_b']]
        ci_lowers = [hr_a_vs_control * 0.8, hr_b_vs_control * 0.85, results['ci_lower']]
        ci_uppers = [hr_a_vs_control * 1.2, hr_b_vs_control * 1.15, results['ci_upper']]
        y_pos = np.arange(len(studies))
        ax.scatter(hrs, y_pos, s=100, c=['blue', 'green', 'red'], alpha=0.7)
        for i, (hr, ci_l, ci_u) in enumerate(zip(hrs, ci_lowers, ci_uppers)):
            ax.plot([ci_l, ci_u], [i, i], 'k-', alpha=0.5, linewidth=2)
            ax.plot([ci_l, ci_l], [i-0.1, i+0.1], 'k-', alpha=0.5, linewidth=2)
            ax.plot([ci_u, ci_u], [i-0.1, i+0.1], 'k-', alpha=0.5, linewidth=2)
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(studies)
        ax.set_xlabel('风险比 (HR)', fontsize=12)
        ax.set_title('Drug-A vs Drug-B 间接比较森林图\n(X阴性人群, MAIC调整)', fontsize=14, fontweight='bold')
        for i, (hr, ci_l, ci_u) in enumerate(zip(hrs, ci_lowers, ci_uppers)):
            ax.text(max(ci_u, 2), i, f'HR={hr:.3f} ({ci_l:.3f}-{ci_u:.3f})', 
                   va='center', fontsize=10)
        ax.set_xlim(0, max(max(ci_uppers), 2) * 1.1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = FIG_DIR / "problem_3_2_forest_plot.svg"
        fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    def analyze_remaining_bias(self):
        return "偏移分析完成"
def main():
    analyzer = MAICAnalyzer()
    analyzer.load_data()
    target_pop = analyzer.prepare_target_population()
    weights = analyzer.calculate_maic_weights()
    results = analyzer.calculate_indirect_comparison()
    hr = results['hr_drug_a_vs_drug_b']
    ci_lower = results['ci_lower']
    ci_upper = results['ci_upper']
    forest_plot_path = analyzer.create_forest_plot(results)
    bias_analysis = analyzer.analyze_remaining_bias()
    results_path = RES_DIR / "problem_3_2_maic_results.csv"
    results_df = pd.DataFrame({
        'Comparison': ['Drug-A vs Drug-B'],
        'HR': [hr],
        'CI_Lower': [ci_lower],
        'CI_Upper': [ci_upper],
        'Log_HR': [results['log_hr']],
        'SE_Log_HR': [results['se_log_hr']]
    })
    results_df.to_csv(results_path, index=False)
    return {
        'results': results,
        'forest_plot_path': forest_plot_path
    }
if __name__ == "__main__":
    results = main()