import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
import zhplot

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('f:\\25DIA\\code1.1\\溶解度数据.csv')

print("=== USP711溶解度统计建模分析 ===")

# 1. 混合效应模型参数估计
print("\n1. 混合效应模型分析")
print("模型: Y_ij = μ + B_i + ε_ij")
print("其中: Y_ij为第i批次第j个药片的溶解度")
print("     μ为总体均值")
print("     B_i ~ N(0, σ²_between)为批次随机效应")
print("     ε_ij ~ N(0, σ²_within)为批次内随机误差")

# 计算各批次均值
batch_means = df.groupby('批次号')['30分钟溶解度'].mean()
overall_mean = df['30分钟溶解度'].mean()

print(f"\n总体均值 (μ): {overall_mean:.3f}%")
print("各批次均值:")
for batch, mean in batch_means.items():
    print(f"  {batch}: {mean:.3f}%")

# 计算批次间方差 (σ²_between)
batch_effects = batch_means - overall_mean
sigma2_between = np.var(batch_effects, ddof=0)  # 使用总体方差公式
print(f"\n批次效应:")
for batch, effect in zip(batch_means.index, batch_effects):
    print(f"  {batch}: {effect:.3f}%")

# 计算批次内方差 (σ²_within)
within_variances = []
for batch in ['B001', 'B002', 'B003']:
    batch_data = df[df['批次号'] == batch]['30分钟溶解度']
    batch_var = np.var(batch_data, ddof=1)  # 样本方差
    within_variances.append(batch_var)
    print(f"批次 {batch} 内方差: {batch_var:.3f}")

sigma2_within = np.mean(within_variances)

print(f"\n方差成分估计:")
print(f"批次间方差 (σ²_between): {sigma2_between:.3f}")
print(f"批次内方差 (σ²_within): {sigma2_within:.3f}")
print(f"总方差: {sigma2_between + sigma2_within:.3f}")

# 计算方差成分比例
total_variance = sigma2_between + sigma2_within
between_ratio = sigma2_between / total_variance
within_ratio = sigma2_within / total_variance

print(f"\n方差成分比例:")
print(f"批次间变异占比: {between_ratio:.1%}")
print(f"批次内变异占比: {within_ratio:.1%}")

# 2. 参数的置信区间估计
print(f"\n2. 参数置信区间估计 (95%置信水平)")

# 总体均值的置信区间
n_total = len(df)
se_mean = np.sqrt(sigma2_within / n_total + sigma2_between / 3)  # 3个批次
t_critical = stats.t.ppf(0.975, df=2)  # 自由度为批次数-1
ci_mean_lower = overall_mean - t_critical * se_mean
ci_mean_upper = overall_mean + t_critical * se_mean

print(f"总体均值 μ: [{ci_mean_lower:.3f}, {ci_mean_upper:.3f}]%")

# 批次内标准差的置信区间 (基于卡方分布)
df_within = 3 * (24 - 1)  # 每批次23个自由度，共3批次
chi2_lower = stats.chi2.ppf(0.025, df_within)
chi2_upper = stats.chi2.ppf(0.975, df_within)

sigma_within_lower = np.sqrt(df_within * sigma2_within / chi2_upper)
sigma_within_upper = np.sqrt(df_within * sigma2_within / chi2_lower)

print(f"批次内标准差 σ_within: [{sigma_within_lower:.3f}, {sigma_within_upper:.3f}]%")

# 3. 模型诊断
print(f"\n3. 模型诊断")

# 计算标准化残差
residuals = []
for batch in ['B001', 'B002', 'B003']:
    batch_data = df[df['批次号'] == batch]['30分钟溶解度']
    batch_mean = batch_means[batch]
    batch_residuals = batch_data - batch_mean
    residuals.extend(batch_residuals)

residuals = np.array(residuals)
standardized_residuals = residuals / np.sqrt(sigma2_within)

# 正态性检验
stat, p_value = stats.shapiro(standardized_residuals)
print(f"残差正态性检验 (Shapiro-Wilk): 统计量={stat:.4f}, p值={p_value:.4f}")

# 4. 预测新批次的分布参数
print(f"\n4. 新批次溶解度分布预测")
print("基于混合效应模型，新批次的溶解度分布:")
print(f"批次均值分布: N({overall_mean:.3f}, {sigma2_between:.3f})")
print(f"批次内分布: N(批次均值, {sigma2_within:.3f})")

# 新批次可能的均值范围 (95%预测区间)
pred_mean_lower = overall_mean - 1.96 * np.sqrt(sigma2_between)
pred_mean_upper = overall_mean + 1.96 * np.sqrt(sigma2_between)
print(f"新批次均值95%预测区间: [{pred_mean_lower:.3f}, {pred_mean_upper:.3f}]%")

# 5. 可视化分析
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 5.1 批次效应图
batch_names = ['B001', 'B002', 'B003']
batch_effects_list = [batch_effects[batch] for batch in batch_names]
colors = ['red' if x < 0 else 'blue' for x in batch_effects_list]

axes[0,0].bar(batch_names, batch_effects_list, color=colors, alpha=0.7)
axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[0,0].set_title('批次随机效应 (B_i)')
axes[0,0].set_xlabel('批次号')
axes[0,0].set_ylabel('批次效应 (%)')
axes[0,0].grid(True, alpha=0.3)

# 5.2 方差成分饼图
labels = ['批次间变异', '批次内变异']
sizes = [sigma2_between, sigma2_within]
colors_pie = ['lightblue', 'lightcoral']

axes[0,1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
axes[0,1].set_title('方差成分分解')

# 5.3 残差分析
axes[1,0].scatter(range(len(standardized_residuals)), standardized_residuals, alpha=0.6)
axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1,0].axhline(y=2, color='orange', linestyle='--', alpha=0.7)
axes[1,0].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
axes[1,0].set_title('标准化残差图')
axes[1,0].set_xlabel('观测序号')
axes[1,0].set_ylabel('标准化残差')
axes[1,0].grid(True, alpha=0.3)

# 5.4 残差Q-Q图
stats.probplot(standardized_residuals, dist="norm", plot=axes[1,1])
axes[1,1].set_title('残差Q-Q图')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('f:\\25DIA\\code1.1\\统计模型分析图.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 保存模型参数
model_params = {
    '总体均值': overall_mean,
    '批次间方差': sigma2_between,
    '批次内方差': sigma2_within,
    '批次间标准差': np.sqrt(sigma2_between),
    '批次内标准差': np.sqrt(sigma2_within),
    '批次间变异占比': between_ratio,
    '批次内变异占比': within_ratio
}

# 保存到CSV文件
params_df = pd.DataFrame(list(model_params.items()), columns=['参数', '估计值'])
params_df.to_csv('f:\\25DIA\\code1.1\\模型参数估计.csv', index=False, encoding='utf-8-sig')

print(f"\n=== 统计建模分析完成 ===")
print("模型参数已保存到 '模型参数估计.csv'")
print("分析图表已保存到 '统计模型分析图.png'")

# 输出关键参数供后续蒙特卡洛模拟使用
print(f"\n=== 关键参数汇总 (供蒙特卡洛模拟使用) ===")
print(f"μ (总体均值): {overall_mean:.6f}")
print(f"σ_between (批次间标准差): {np.sqrt(sigma2_between):.6f}")
print(f"σ_within (批次内标准差): {np.sqrt(sigma2_within):.6f}")