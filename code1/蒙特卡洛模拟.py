"""
蒙特卡洛模拟估算USP711各阶段失败率
基于现有批次数据的统计分布进行模拟
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from USP711测试逻辑 import USP711Tester
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MonteCarloSimulator:
    def __init__(self, data_file='溶解度数据.csv'):
        """初始化模拟器"""
        self.data = pd.read_csv(data_file)
        self.tester = USP711Tester()
        self.batch_params = {}
        self._estimate_distribution_parameters()
    
    def _estimate_distribution_parameters(self):
        """估计每个批次的分布参数"""
        print("=== 分布参数估计 ===")
        
        for batch in self.data['batch_id'].unique():
            batch_data = self.data[self.data['batch_id'] == batch]['dissolution'].values
            
            # 正态分布参数估计
            mu, sigma = stats.norm.fit(batch_data)
            
            # 正态性检验
            shapiro_stat, shapiro_p = stats.shapiro(batch_data)
            
            # 存储参数
            self.batch_params[batch] = {
                'mean': mu,
                'std': sigma,
                'n_samples': len(batch_data),
                'shapiro_p': shapiro_p,
                'raw_data': batch_data
            }
            
            print(f"批次 {batch}:")
            print(f"  均值: {mu:.2f}%")
            print(f"  标准差: {sigma:.2f}%")
            print(f"  样本数: {len(batch_data)}")
            print(f"  正态性检验p值: {shapiro_p:.4f}")
            print(f"  正态性: {'是' if shapiro_p > 0.05 else '否'}")
            print()
    
    def simulate_batch(self, mean, std, n_tablets=24, random_state=None):
        """模拟一个批次的溶解度数据"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # 生成正态分布的溶解度数据
        dissolution_values = np.random.normal(mean, std, n_tablets)
        
        # 确保溶解度在合理范围内 (0-100%)
        dissolution_values = np.clip(dissolution_values, 0, 100)
        
        return dissolution_values
    
    def monte_carlo_simulation(self, n_simulations=10000, batch_type='pooled'):
        """
        蒙特卡洛模拟
        
        Parameters:
        - n_simulations: 模拟次数
        - batch_type: 'pooled' (合并所有批次), 'individual' (分别模拟各批次), 或具体批次名
        """
        print(f"=== 蒙特卡洛模拟 ({n_simulations}次) ===")
        
        if batch_type == 'pooled':
            # 合并所有批次数据估计参数
            all_data = self.data['dissolution'].values
            mean, std = stats.norm.fit(all_data)
            print(f"合并数据参数 - 均值: {mean:.2f}%, 标准差: {std:.2f}%")
        elif batch_type == 'individual':
            # 分别对每个批次进行模拟
            results = {}
            for batch in self.batch_params.keys():
                print(f"\n--- 批次 {batch} 模拟 ---")
                batch_results = self._run_simulation(
                    self.batch_params[batch]['mean'],
                    self.batch_params[batch]['std'],
                    n_simulations
                )
                results[batch] = batch_results
            return results
        else:
            # 指定批次模拟
            if batch_type in self.batch_params:
                mean = self.batch_params[batch_type]['mean']
                std = self.batch_params[batch_type]['std']
                print(f"批次 {batch_type} 参数 - 均值: {mean:.2f}%, 标准差: {std:.2f}%")
            else:
                raise ValueError(f"未找到批次 {batch_type}")
        
        return self._run_simulation(mean, std, n_simulations)
    
    def _run_simulation(self, mean, std, n_simulations):
        """执行模拟"""
        results = {
            'stage1_pass': 0,
            'stage2_pass': 0,
            'stage3_pass': 0,
            'overall_pass': 0,
            'failure_stages': [],
            'stage_failure_details': {1: [], 2: [], 3: []}
        }
        
        for i in range(n_simulations):
            # 生成模拟批次
            dissolution_data = self.simulate_batch(mean, std, random_state=i)
            
            # 进行USP711测试
            test_result, failure_stage, details = self.tester.full_test(dissolution_data.tolist())
            
            # 分析各阶段通过情况
            # 重新测试各阶段以获得详细信息
            available_tablets = dissolution_data.tolist()
            np.random.seed(i)
            np.random.shuffle(available_tablets)
            
            # 第一阶段测试
            stage1_pass, _ = self.tester.stage1_test(available_tablets[:6])
            if stage1_pass:
                results['stage1_pass'] += 1
            
            # 第二阶段测试
            stage2_pass, _ = self.tester.stage2_test(available_tablets[:12])
            if stage2_pass:
                results['stage2_pass'] += 1
            
            # 第三阶段测试
            stage3_pass, _ = self.tester.stage3_test(available_tablets[:24])
            if stage3_pass:
                results['stage3_pass'] += 1
            
            # 整体测试结果
            if test_result == "通过":
                results['overall_pass'] += 1
            else:
                if failure_stage > 0:
                    results['failure_stages'].append(failure_stage)
        
        # 计算失败率
        failure_rates = {
            'stage1_failure_rate': 1 - results['stage1_pass'] / n_simulations,
            'stage2_failure_rate': 1 - results['stage2_pass'] / n_simulations,
            'stage3_failure_rate': 1 - results['stage3_pass'] / n_simulations,
            'overall_failure_rate': 1 - results['overall_pass'] / n_simulations
        }
        
        # 计算置信区间
        confidence_intervals = self._calculate_confidence_intervals(results, n_simulations)
        
        # 打印结果
        self._print_simulation_results(failure_rates, confidence_intervals, n_simulations)
        
        return {
            'failure_rates': failure_rates,
            'confidence_intervals': confidence_intervals,
            'raw_results': results,
            'n_simulations': n_simulations
        }
    
    def _calculate_confidence_intervals(self, results, n_simulations, confidence=0.95):
        """计算失败率的置信区间"""
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        intervals = {}
        for stage in ['stage1', 'stage2', 'stage3', 'overall']:
            pass_count = results[f'{stage}_pass']
            failure_rate = 1 - pass_count / n_simulations
            
            # 二项分布的置信区间
            se = np.sqrt(failure_rate * (1 - failure_rate) / n_simulations)
            margin = z_score * se
            
            intervals[f'{stage}_failure_rate'] = {
                'lower': max(0, failure_rate - margin),
                'upper': min(1, failure_rate + margin)
            }
        
        return intervals
    
    def _print_simulation_results(self, failure_rates, confidence_intervals, n_simulations):
        """打印模拟结果"""
        print(f"\n模拟次数: {n_simulations}")
        print("\n各阶段失败率估计:")
        
        stages = [
            ('第一阶段', 'stage1_failure_rate'),
            ('第二阶段', 'stage2_failure_rate'), 
            ('第三阶段', 'stage3_failure_rate'),
            ('整体测试', 'overall_failure_rate')
        ]
        
        for stage_name, key in stages:
            rate = failure_rates[key]
            ci = confidence_intervals[key]
            print(f"  {stage_name}: {rate:.4f} ({rate*100:.2f}%) "
                  f"[95%CI: {ci['lower']:.4f}-{ci['upper']:.4f}]")
    
    def analyze_failure_patterns(self, simulation_results):
        """分析失败模式"""
        print("\n=== 失败模式分析 ===")
        
        failure_stages = simulation_results['raw_results']['failure_stages']
        if failure_stages:
            stage_counts = pd.Series(failure_stages).value_counts().sort_index()
            print("失败阶段分布:")
            for stage, count in stage_counts.items():
                percentage = count / len(failure_stages) * 100
                print(f"  第{stage}阶段失败: {count}次 ({percentage:.1f}%)")
        else:
            print("所有模拟批次均通过测试")
    
    def create_visualizations(self, simulation_results, save_plots=True):
        """创建可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('USP711溶解度测试失败率分析', fontsize=16, fontweight='bold')
        
        # 1. 各阶段失败率柱状图
        failure_rates = simulation_results['failure_rates']
        stages = ['第一阶段', '第二阶段', '第三阶段', '整体测试']
        rates = [failure_rates['stage1_failure_rate'], 
                failure_rates['stage2_failure_rate'],
                failure_rates['stage3_failure_rate'], 
                failure_rates['overall_failure_rate']]
        
        bars = axes[0,0].bar(stages, rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0,0].set_title('各阶段失败率')
        axes[0,0].set_ylabel('失败率')
        axes[0,0].set_ylim(0, max(rates) * 1.2)
        
        # 添加数值标签
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + max(rates)*0.01,
                          f'{rate:.4f}\n({rate*100:.2f}%)', 
                          ha='center', va='bottom', fontsize=10)
        
        # 2. 失败阶段分布饼图
        failure_stages = simulation_results['raw_results']['failure_stages']
        if failure_stages:
            stage_counts = pd.Series(failure_stages).value_counts().sort_index()
            labels = [f'第{stage}阶段' for stage in stage_counts.index]
            axes[0,1].pie(stage_counts.values, labels=labels, autopct='%1.1f%%', 
                         colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0,1].set_title('失败阶段分布')
        else:
            axes[0,1].text(0.5, 0.5, '所有批次均通过测试', ha='center', va='center', 
                          transform=axes[0,1].transAxes, fontsize=14)
            axes[0,1].set_title('失败阶段分布')
        
        # 3. 置信区间图
        ci = simulation_results['confidence_intervals']
        y_pos = np.arange(len(stages))
        
        for i, (stage, rate) in enumerate(zip(stages, rates)):
            ci_key = ['stage1_failure_rate', 'stage2_failure_rate', 
                     'stage3_failure_rate', 'overall_failure_rate'][i]
            lower = ci[ci_key]['lower']
            upper = ci[ci_key]['upper']
            
            axes[1,0].barh(i, rate, color=bars[i].get_facecolor(), alpha=0.7)
            axes[1,0].errorbar(rate, i, xerr=[[rate-lower], [upper-rate]], 
                              fmt='o', color='black', capsize=5)
        
        axes[1,0].set_yticks(y_pos)
        axes[1,0].set_yticklabels(stages)
        axes[1,0].set_xlabel('失败率')
        axes[1,0].set_title('失败率及95%置信区间')
        
        # 4. 批次数据分布对比
        axes[1,1].hist([params['raw_data'] for params in self.batch_params.values()], 
                      bins=15, alpha=0.7, label=list(self.batch_params.keys()),
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1,1].set_xlabel('溶解度 (%)')
        axes[1,1].set_ylabel('频数')
        axes[1,1].set_title('各批次溶解度分布')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('USP711失败分析图.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'USP711失败分析图.png'")
        
        plt.show()
        
        return fig

def main():
    """主函数"""
    print("USP711溶解度测试失败率蒙特卡洛模拟分析")
    print("=" * 50)
    
    # 初始化模拟器
    simulator = MonteCarloSimulator()
    
    # 进行蒙特卡洛模拟
    print("\n1. 合并数据模拟")
    pooled_results = simulator.monte_carlo_simulation(n_simulations=10000, batch_type='pooled')
    
    print("\n2. 各批次单独模拟")
    individual_results = simulator.monte_carlo_simulation(n_simulations=10000, batch_type='individual')
    
    # 分析失败模式
    simulator.analyze_failure_patterns(pooled_results)
    
    # 生成可视化
    simulator.create_visualizations(pooled_results)
    
    # 保存结果
    results_summary = {
        'pooled_simulation': pooled_results['failure_rates'],
        'individual_simulations': {batch: results['failure_rates'] 
                                 for batch, results in individual_results.items()},
        'confidence_intervals': pooled_results['confidence_intervals']
    }
    
    # 保存到CSV
    summary_df = pd.DataFrame(results_summary['pooled_simulation'], index=[0])
    summary_df.to_csv('蒙特卡洛结果.csv', index=False)
    print("\n结果已保存到 '蒙特卡洛结果.csv'")
    
    return results_summary

if __name__ == "__main__":
    results = main()