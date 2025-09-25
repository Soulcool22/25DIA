import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import zhplot
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重现
np.random.seed(42)

print("=== USP711溶解度失败率蒙特卡洛模拟 ===")

# 从统计建模结果中获取参数
mu = 82.153  # 总体均值
sigma_between = np.sqrt(2.561)  # 批次间标准差
sigma_within = np.sqrt(6.652)   # 批次内标准差

print(f"模型参数:")
print(f"总体均值 (μ): {mu:.3f}%")
print(f"批次间标准差 (σ_between): {sigma_between:.3f}%")
print(f"批次内标准差 (σ_within): {sigma_within:.3f}%")

# USP711测试标准
Q = 80  # 标准值
Q_plus_5 = 85  # Q+5%
Q_minus_15 = 65  # Q-15%
Q_minus_25 = 55  # Q-25%

print(f"\nUSP711测试标准:")
print(f"Q = {Q}%")
print(f"Q+5% = {Q_plus_5}% (第一阶段通过标准)")
print(f"Q-15% = {Q_minus_15}% (第二、三阶段最低标准)")
print(f"Q-25% = {Q_minus_25}% (第三阶段绝对最低标准)")

def usp711_test(dissolution_values):
    """
    执行USP711三阶段测试
    返回: (通过阶段, 是否通过, 失败原因)
    """
    n = len(dissolution_values)
    
    if n >= 6:
        # 第一阶段：前6个样品
        stage1_samples = dissolution_values[:6]
        if all(x >= Q_plus_5 for x in stage1_samples):
            return 1, True, "第一阶段通过"
    
    if n >= 12:
        # 第二阶段：前12个样品
        stage2_samples = dissolution_values[:12]
        stage2_mean = np.mean(stage2_samples)
        stage2_below_q_minus_15 = sum(1 for x in stage2_samples if x < Q_minus_15)
        
        if stage2_mean >= Q and stage2_below_q_minus_15 == 0:
            return 2, True, "第二阶段通过"
    
    if n >= 24:
        # 第三阶段：全部24个样品
        stage3_samples = dissolution_values[:24]
        stage3_mean = np.mean(stage3_samples)
        stage3_below_q_minus_15 = sum(1 for x in stage3_samples if x < Q_minus_15)
        stage3_below_q_minus_25 = sum(1 for x in stage3_samples if x < Q_minus_25)
        
        if (stage3_mean >= Q and 
            stage3_below_q_minus_15 <= 2 and 
            stage3_below_q_minus_25 == 0):
            return 3, True, "第三阶段通过"
        else:
            # 确定失败原因
            if stage3_mean < Q:
                return 3, False, f"第三阶段失败：平均值{stage3_mean:.1f}% < {Q}%"
            elif stage3_below_q_minus_15 > 2:
                return 3, False, f"第三阶段失败：{stage3_below_q_minus_15}片 < {Q_minus_15}% (超过2片)"
            elif stage3_below_q_minus_25 > 0:
                return 3, False, f"第三阶段失败：{stage3_below_q_minus_25}片 < {Q_minus_25}%"
    
    return 3, False, "未知失败原因"

def simulate_batch():
    """模拟一个批次的溶解度数据"""
    # 生成批次均值
    batch_mean = np.random.normal(mu, sigma_between)
    
    # 生成24个药片的溶解度值
    dissolution_values = np.random.normal(batch_mean, sigma_within, 24)
    
    return dissolution_values

# 蒙特卡洛模拟
n_simulations = 100000
print(f"\n开始蒙特卡洛模拟 (模拟次数: {n_simulations:,})")

# 存储结果
results = {
    'stage1_pass': 0,
    'stage2_pass': 0,
    'stage3_pass': 0,
    'total_fail': 0,
    'stage1_fail': 0,
    'stage2_fail': 0,
    'stage3_fail': 0,
    'failure_reasons': []
}

# 详细统计
stage_results = []
batch_means = []
failure_details = {
    'stage3_mean_fail': 0,
    'stage3_below_65_fail': 0,
    'stage3_below_55_fail': 0
}

# 进行模拟
for i in tqdm(range(n_simulations), desc="模拟进度"):
    dissolution_values = simulate_batch()
    batch_means.append(np.mean(dissolution_values))
    
    stage, passed, reason = usp711_test(dissolution_values)
    
    stage_results.append({
        'simulation': i+1,
        'batch_mean': np.mean(dissolution_values),
        'stage': stage,
        'passed': passed,
        'reason': reason
    })
    
    if passed:
        if stage == 1:
            results['stage1_pass'] += 1
        elif stage == 2:
            results['stage2_pass'] += 1
        elif stage == 3:
            results['stage3_pass'] += 1
    else:
        results['total_fail'] += 1
        results['failure_reasons'].append(reason)
        
        if stage == 1:
            results['stage1_fail'] += 1
        elif stage == 2:
            results['stage2_fail'] += 1
        elif stage == 3:
            results['stage3_fail'] += 1
            
            # 详细分析第三阶段失败原因
            if "平均值" in reason:
                failure_details['stage3_mean_fail'] += 1
            elif "片 < 65%" in reason:
                failure_details['stage3_below_65_fail'] += 1
            elif "片 < 55%" in reason:
                failure_details['stage3_below_55_fail'] += 1

# 计算失败率
print(f"\n=== 蒙特卡洛模拟结果 ===")
print(f"总模拟次数: {n_simulations:,}")

# 各阶段通过率
stage1_pass_rate = results['stage1_pass'] / n_simulations
stage2_pass_rate = results['stage2_pass'] / n_simulations
stage3_pass_rate = results['stage3_pass'] / n_simulations
total_pass_rate = (results['stage1_pass'] + results['stage2_pass'] + results['stage3_pass']) / n_simulations

print(f"\n各阶段通过情况:")
print(f"第一阶段通过: {results['stage1_pass']:,} 次 ({stage1_pass_rate:.2%})")
print(f"第二阶段通过: {results['stage2_pass']:,} 次 ({stage2_pass_rate:.2%})")
print(f"第三阶段通过: {results['stage3_pass']:,} 次 ({stage3_pass_rate:.2%})")
print(f"总通过率: {total_pass_rate:.2%}")

# 各阶段失败率
stage1_fail_rate = results['stage1_fail'] / n_simulations
stage2_fail_rate = results['stage2_fail'] / n_simulations
stage3_fail_rate = results['stage3_fail'] / n_simulations
total_fail_rate = results['total_fail'] / n_simulations

print(f"\n各阶段失败情况:")
print(f"第一阶段失败: {results['stage1_fail']:,} 次 ({stage1_fail_rate:.2%})")
print(f"第二阶段失败: {results['stage2_fail']:,} 次 ({stage2_fail_rate:.2%})")
print(f"第三阶段失败: {results['stage3_fail']:,} 次 ({stage3_fail_rate:.2%})")
print(f"总失败率: {total_fail_rate:.2%}")

# 第三阶段失败原因详细分析
print(f"\n第三阶段失败原因分析:")
if results['stage3_fail'] > 0:
    mean_fail_pct = failure_details['stage3_mean_fail'] / results['stage3_fail']
    below_65_fail_pct = failure_details['stage3_below_65_fail'] / results['stage3_fail']
    below_55_fail_pct = failure_details['stage3_below_55_fail'] / results['stage3_fail']
    
    print(f"平均值不达标: {failure_details['stage3_mean_fail']:,} 次 ({mean_fail_pct:.1%})")
    print(f"超过2片<65%: {failure_details['stage3_below_65_fail']:,} 次 ({below_65_fail_pct:.1%})")
    print(f"存在<55%药片: {failure_details['stage3_below_55_fail']:,} 次 ({below_55_fail_pct:.1%})")

# 置信区间计算
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)

def calculate_ci(success_count, total_count):
    p = success_count / total_count
    se = np.sqrt(p * (1 - p) / total_count)
    margin = z_score * se
    return p - margin, p + margin

print(f"\n失败率95%置信区间:")
total_fail_ci = calculate_ci(results['total_fail'], n_simulations)
stage1_fail_ci = calculate_ci(results['stage1_fail'], n_simulations)
stage2_fail_ci = calculate_ci(results['stage2_fail'], n_simulations)
stage3_fail_ci = calculate_ci(results['stage3_fail'], n_simulations)

print(f"总失败率: {total_fail_rate:.3%} [{total_fail_ci[0]:.3%}, {total_fail_ci[1]:.3%}]")
print(f"第一阶段失败率: {stage1_fail_rate:.3%} [{stage1_fail_ci[0]:.3%}, {stage1_fail_ci[1]:.3%}]")
print(f"第二阶段失败率: {stage2_fail_rate:.3%} [{stage2_fail_ci[0]:.3%}, {stage2_fail_ci[1]:.3%}]")
print(f"第三阶段失败率: {stage3_fail_rate:.3%} [{stage3_fail_ci[0]:.3%}, {stage3_fail_ci[1]:.3%}]")

# 保存详细结果
results_df = pd.DataFrame(stage_results)
results_df.to_csv('f:\\25DIA\\code1.1\\蒙特卡洛结果.csv', index=False, encoding='utf-8-sig')

# 汇总统计
summary_stats = {
    '指标': ['总失败率', '第一阶段失败率', '第二阶段失败率', '第三阶段失败率',
            '第一阶段通过率', '第二阶段通过率', '第三阶段通过率', '总通过率'],
    '估计值': [f"{total_fail_rate:.3%}", f"{stage1_fail_rate:.3%}", f"{stage2_fail_rate:.3%}", f"{stage3_fail_rate:.3%}",
              f"{stage1_pass_rate:.3%}", f"{stage2_pass_rate:.3%}", f"{stage3_pass_rate:.3%}", f"{total_pass_rate:.3%}"],
    '95%置信区间下限': [f"{total_fail_ci[0]:.3%}", f"{stage1_fail_ci[0]:.3%}", f"{stage2_fail_ci[0]:.3%}", f"{stage3_fail_ci[0]:.3%}",
                    f"{1-total_fail_ci[1]:.3%}", f"{stage1_pass_rate-z_score*np.sqrt(stage1_pass_rate*(1-stage1_pass_rate)/n_simulations):.3%}",
                    f"{stage2_pass_rate-z_score*np.sqrt(stage2_pass_rate*(1-stage2_pass_rate)/n_simulations):.3%}",
                    f"{total_pass_rate-z_score*np.sqrt(total_pass_rate*(1-total_pass_rate)/n_simulations):.3%}"],
    '95%置信区间上限': [f"{total_fail_ci[1]:.3%}", f"{stage1_fail_ci[1]:.3%}", f"{stage2_fail_ci[1]:.3%}", f"{stage3_fail_ci[1]:.3%}",
                    f"{1-total_fail_ci[0]:.3%}", f"{stage1_pass_rate+z_score*np.sqrt(stage1_pass_rate*(1-stage1_pass_rate)/n_simulations):.3%}",
                    f"{stage2_pass_rate+z_score*np.sqrt(stage2_pass_rate*(1-stage2_pass_rate)/n_simulations):.3%}",
                    f"{total_pass_rate+z_score*np.sqrt(total_pass_rate*(1-total_pass_rate)/n_simulations):.3%}"]
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('f:\\25DIA\\code1.1\\失败率汇总统计.csv', index=False, encoding='utf-8-sig')

# 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 失败率条形图
categories = ['第一阶段', '第二阶段', '第三阶段', '总体']
fail_rates = [stage1_fail_rate, stage2_fail_rate, stage3_fail_rate, total_fail_rate]
colors = ['red', 'orange', 'blue', 'purple']

bars = axes[0,0].bar(categories, fail_rates, color=colors, alpha=0.7)
axes[0,0].set_title('各阶段失败率')
axes[0,0].set_ylabel('失败率')
axes[0,0].set_ylim(0, max(fail_rates) * 1.2)

# 添加数值标签
for bar, rate in zip(bars, fail_rates):
    height = bar.get_height()
    axes[0,0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{rate:.2%}', ha='center', va='bottom')

# 2. 批次均值分布
axes[0,1].hist(batch_means, bins=50, alpha=0.7, color='skyblue', density=True)
axes[0,1].axvline(mu, color='red', linestyle='--', label=f'理论均值 {mu:.1f}%')
axes[0,1].axvline(np.mean(batch_means), color='orange', linestyle='--', label=f'模拟均值 {np.mean(batch_means):.1f}%')
axes[0,1].set_title('模拟批次均值分布')
axes[0,1].set_xlabel('批次均值 (%)')
axes[0,1].set_ylabel('密度')
axes[0,1].legend()

# 3. 通过率饼图
pass_counts = [results['stage1_pass'], results['stage2_pass'], results['stage3_pass'], results['total_fail']]
pass_labels = ['第一阶段通过', '第二阶段通过', '第三阶段通过', '失败']
pass_colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral']

axes[1,0].pie(pass_counts, labels=pass_labels, colors=pass_colors, autopct='%1.1f%%', startangle=90)
axes[1,0].set_title('测试结果分布')

# 4. 第三阶段失败原因
if results['stage3_fail'] > 0:
    failure_reasons = ['平均值不达标', '超过2片<65%', '存在<55%药片']
    failure_counts = [failure_details['stage3_mean_fail'], 
                     failure_details['stage3_below_65_fail'], 
                     failure_details['stage3_below_55_fail']]
    
    axes[1,1].bar(failure_reasons, failure_counts, color=['red', 'orange', 'purple'], alpha=0.7)
    axes[1,1].set_title('第三阶段失败原因分析')
    axes[1,1].set_ylabel('失败次数')
    axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('f:\\25DIA\\code1.1\\蒙特卡洛模拟结果图.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== 蒙特卡洛模拟完成 ===")
print("详细结果已保存到 '蒙特卡洛结果.csv'")
print("汇总统计已保存到 '失败率汇总统计.csv'")
print("结果图表已保存到 '蒙特卡洛模拟结果图.png'")

# 最终结论
print(f"\n=== 最终结论 ===")
print(f"基于{n_simulations:,}次蒙特卡洛模拟，该药品未来批次的USP711溶解度测试:")
print(f"• 总失败率: {total_fail_rate:.2%} (95%CI: {total_fail_ci[0]:.2%}-{total_fail_ci[1]:.2%})")
print(f"• 第一阶段失败率: {stage1_fail_rate:.2%}")
print(f"• 第二阶段失败率: {stage2_fail_rate:.2%}")
print(f"• 第三阶段失败率: {stage3_fail_rate:.2%}")
print(f"• 总通过率: {total_pass_rate:.2%}")