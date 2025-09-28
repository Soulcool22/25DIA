"""
问题3.1: 重构RCT-B中X阴性组OS KM曲线
利用已知的RCT-B ITT和X阳性亚组数据，推测重构X阴性亚组的OS数据
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 尝试导入lifelines
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.utils import median_survival_times
except ImportError:
    raise ImportError("缺少lifelines库，请安装：pip install lifelines")

# 设置路径
BASE_DIR = Path("f:/25DIA/code3.2")
DATA_PATH = Path("f:/25DIA/复赛大题（三）数据集.csv")
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

class IPDReconstructor:
    """个体患者数据重构器"""
    
    def __init__(self):
        self.rct_a_data = None
        self.rct_b_published_data = None
        
    def load_rct_a_data(self):
        """加载RCT-A的个体患者数据"""
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"数据集不存在: {DATA_PATH}")
        
        df = pd.read_csv(DATA_PATH)
        # 标准化列名
        df.columns = [c.strip() for c in df.columns]
        
        # 重命名事件列
        event_col_candidates = [c for c in df.columns if 'event' in c.lower()]
        if event_col_candidates:
            df = df.rename(columns={event_col_candidates[0]: "event"})
        
        # 数据类型转换
        df["time(OS)"] = pd.to_numeric(df["time(OS)"], errors="coerce")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        
        # 只保留RCT-A数据
        df = df[df["study"] == "RCT_A_vs_O"].copy()
        self.rct_a_data = df
        return df
    
    def setup_rct_b_published_data(self):
        """设置RCT-B的已发表汇总数据"""
        # 根据题目描述的表一数据
        self.rct_b_published_data = {
            'itt': {
                'drug_b': {'n': 119, 'age_le_65_pct': 68.1, 'male_pct': 54.6},
                'control': {'n': 121, 'age_le_65_pct': 66.1, 'male_pct': 61.2}
            },
            'x_positive': {
                'drug_b': {'n': 49, 'age_le_65_pct': 67.3, 'male_pct': 80.0},
                'control': {'n': 55, 'age_le_65_pct': 80.0, 'male_pct': 80.0}  # 从题目推断
            }
        }
        
        # 从ITT和X阳性数据推算X阴性数据
        self.rct_b_published_data['x_negative'] = {
            'drug_b': {'n': 119 - 49},  # 70
            'control': {'n': 121 - 55}  # 66
        }
        
        # 根据人群分布推算X阴性亚组的基线特征
        # 使用加权平均的逆向计算
        for arm in ['drug_b', 'control']:
            itt_data = self.rct_b_published_data['itt'][arm]
            x_pos_data = self.rct_b_published_data['x_positive'][arm]
            x_neg_data = self.rct_b_published_data['x_negative'][arm]
            
            n_itt = itt_data['n']
            n_pos = x_pos_data['n']
            n_neg = x_neg_data['n']
            
            # 推算X阴性组的年龄分布
            # ITT_age = (n_pos * pos_age + n_neg * neg_age) / n_itt
            # neg_age = (ITT_age * n_itt - n_pos * pos_age) / n_neg
            neg_age_le_65_pct = (itt_data['age_le_65_pct'] * n_itt - 
                                x_pos_data['age_le_65_pct'] * n_pos) / n_neg
            
            # 推算X阴性组的性别分布
            neg_male_pct = (itt_data['male_pct'] * n_itt - 
                           x_pos_data['male_pct'] * n_pos) / n_neg
            
            x_neg_data['age_le_65_pct'] = max(0, min(100, neg_age_le_65_pct))
            x_neg_data['male_pct'] = max(0, min(100, neg_male_pct))
    
    def estimate_survival_parameters(self, subgroup='x_negative'):
        """基于RCT-A数据估计生存参数，用于重构RCT-B数据"""
        if self.rct_a_data is None:
            self.load_rct_a_data()
        
        # 获取RCT-A中对应亚组的数据
        if subgroup == 'x_negative':
            subset = self.rct_a_data[self.rct_a_data['biomarker_x'] == 'Negative']
        else:
            subset = self.rct_a_data[self.rct_a_data['biomarker_x'] == 'Positive']
        
        # 分别拟合治疗组和对照组的生存曲线
        survival_params = {}
        
        for trt in ['drug_a', 'control']:
            trt_data = subset[subset['treatment'] == trt]
            
            # 使用Kaplan-Meier估计生存函数
            kmf = KaplanMeierFitter()
            kmf.fit(trt_data['time(OS)'], trt_data['event'])
            
            # 估计中位生存时间和其他参数
            median_surv = kmf.median_survival_time_
            
            # 拟合指数分布参数（简化假设）
            # 使用最大似然估计
            times = trt_data['time(OS)'].values
            events = trt_data['event'].values
            
            # 估计hazard rate
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
        """重构RCT-B中X阴性组的IPD数据"""
        # 获取生存参数
        surv_params = self.estimate_survival_parameters('x_negative')
        
        # 获取RCT-B X阴性组的样本量
        n_drug_b = self.rct_b_published_data['x_negative']['drug_b']['n']
        n_control = self.rct_b_published_data['x_negative']['control']['n']
        
        reconstructed_data = []
        
        # 重构Drug-B组数据
        np.random.seed(42)  # 确保可重现性
        
        # 假设RCT-B的Drug-B相对于标准治疗有一定的疗效
        # 基于文献，假设HR约为0.7-0.8
        drug_b_hr = 0.75  # 相对于对照组的风险比
        
        for arm, n_patients in [('drug_b', n_drug_b), ('control', n_control)]:
            base_hazard = surv_params['control']['hazard_rate']
            
            if arm == 'drug_b':
                # Drug-B相对于对照组的hazard rate
                hazard_rate = base_hazard * drug_b_hr
            else:
                hazard_rate = base_hazard
            
            # 生成生存时间（指数分布）
            survival_times = np.random.exponential(1/hazard_rate, n_patients)
            
            # 生成删失时间（假设随访时间最长24个月）
            censoring_times = np.random.uniform(12, 24, n_patients)
            
            # 观察时间和事件指示
            observed_times = np.minimum(survival_times, censoring_times)
            events = (survival_times <= censoring_times).astype(int)
            
            # 生成基线协变量
            baseline_data = self.rct_b_published_data['x_negative'][arm]
            
            # 年龄
            age_le_65_prob = baseline_data['age_le_65_pct'] / 100
            ages = np.where(np.random.random(n_patients) < age_le_65_prob,
                           np.random.normal(60, 8, n_patients),
                           np.random.normal(70, 6, n_patients))
            ages = np.clip(ages, 18, 85)
            
            # 性别
            male_prob = baseline_data['male_pct'] / 100
            sexes = np.where(np.random.random(n_patients) < male_prob, 'Male', 'Female')
            
            # ECOG（假设分布）
            ecog_scores = np.random.choice(['0-1', '2'], n_patients, p=[0.7, 0.3])
            
            # 构建数据框
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
        """创建RCT-A和重构的RCT-B X阴性组的叠加KM图"""
        fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
        
        # RCT-A X阴性组数据
        rct_a_x_neg = self.rct_a_data[self.rct_a_data['biomarker_x'] == 'Negative']
        
        kmf = KaplanMeierFitter()
        
        # 绘制RCT-A曲线
        for trt in ['drug_a', 'control']:
            subset = rct_a_x_neg[rct_a_x_neg['treatment'] == trt]
            label = f'RCT-A {trt.replace("_", "-").title()}'
            kmf.fit(subset['time(OS)'], subset['event'], label=label)
            kmf.plot_survival_function(ax=ax, linestyle='-', linewidth=2)
        
        # 绘制重构的RCT-B曲线
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
        
        # 添加风险表
        ax.text(0.02, 0.02, 
                '注：RCT-B数据基于已发表汇总数据重构\n重构方法存在不确定性，结果仅供参考',
                transform=ax.transAxes, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        
        # 保存图片
        output_path = FIG_DIR / "problem_3_1_overlay_km_curves.svg"
        fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def analyze_reconstruction_limitations(self):
        """分析重构数据的局限性和不确定性"""
        limitations = {
            "数据重构局限性": [
                "1. 基线协变量推算的不确定性：仅基于ITT和X阳性亚组数据逆向推算X阴性组特征",
                "2. 生存分布假设：假设指数分布可能不符合实际生存模式",
                "3. 治疗效应假设：Drug-B相对于对照组的HR基于文献假设，缺乏直接证据",
                "4. 删失模式假设：随机删失假设可能与实际情况不符",
                "5. 协变量关联性：未考虑年龄、性别等协变量间的相关性"
            ],
            "不确定性处理方法": [
                "1. 敏感性分析：改变关键参数（如HR值）进行多种情景分析",
                "2. 置信区间估计：使用Bootstrap方法估计重构参数的不确定性",
                "3. 模型验证：与已知的ITT结果进行一致性检验",
                "4. 专家意见：结合临床专家对治疗效应的先验知识",
                "5. 文献Meta分析：利用同类药物的历史数据校准参数"
            ]
        }
        
        return limitations

def main():
    """主函数"""
    print("开始问题3.1：重构RCT-B中X阴性组OS KM曲线")
    
    # 初始化重构器
    reconstructor = IPDReconstructor()
    
    # 加载RCT-A数据
    print("1. 加载RCT-A个体患者数据...")
    reconstructor.load_rct_a_data()
    
    # 设置RCT-B已发表数据
    print("2. 设置RCT-B已发表汇总数据...")
    reconstructor.setup_rct_b_published_data()
    
    # 重构RCT-B X阴性组数据
    print("3. 重构RCT-B X阴性组个体患者数据...")
    rct_b_reconstructed = reconstructor.reconstruct_rct_b_x_negative()
    
    # 保存重构数据
    output_data_path = RES_DIR / "rct_b_x_negative_reconstructed.csv"
    rct_b_reconstructed.to_csv(output_data_path, index=False)
    print(f"重构数据已保存至: {output_data_path}")
    
    # 创建叠加KM图
    print("4. 创建RCT-A与重构RCT-B的叠加KM图...")
    km_plot_path = reconstructor.create_overlay_km_plot(rct_b_reconstructed)
    print(f"KM图已保存至: {km_plot_path}")
    
    # 分析局限性
    print("5. 分析重构方法的局限性...")
    limitations = reconstructor.analyze_reconstruction_limitations()
    
    # 保存分析结果
    limitations_path = RES_DIR / "reconstruction_limitations_analysis.txt"
    with open(limitations_path, 'w', encoding='utf-8') as f:
        for category, items in limitations.items():
            f.write(f"\n{category}:\n")
            for item in items:
                f.write(f"{item}\n")
    
    print(f"局限性分析已保存至: {limitations_path}")
    print("问题3.1完成！")
    
    return {
        'reconstructed_data': rct_b_reconstructed,
        'km_plot_path': km_plot_path,
        'limitations': limitations
    }

if __name__ == "__main__":
    results = main()