import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import zhplot

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建溶解度数据
data = {
    '批次号': ['B001'] * 24 + ['B002'] * 24 + ['B003'] * 24,
    '药片': list(range(1, 25)) * 3,
    '30分钟溶解度': [
        # B001批次
        85, 83, 87, 81, 86, 84, 85, 84, 85, 82, 83, 84, 81, 85, 87, 78, 86, 84, 88, 84, 80, 83, 85, 82,
        # B002批次
        80, 82, 80, 81, 83, 78, 81, 81, 81, 79, 82, 80, 79, 81, 80, 83, 76, 75, 80, 79, 81, 78, 82, 78,
        # B003批次
        80, 85, 79, 82, 77, 75, 88, 83, 84, 78, 86, 83, 87, 86, 84, 81, 82, 83, 86, 84, 85, 81, 81, 83
    ]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 保存原始数据
df.to_csv('f:\\25DIA\\code1.1\\溶解度数据.csv', index=False, encoding='utf-8-sig')

print("=== 溶解度数据描述性统计分析 ===")
print("\n1. 整体数据概览:")
print(df.describe())

print("\n2. 各批次描述性统计:")
batch_stats = df.groupby('批次号')['30分钟溶解度'].describe()
print(batch_stats)

# 计算各批次的统计指标
print("\n3. 各批次详细统计指标:")
for batch in ['B001', 'B002', 'B003']:
    batch_data = df[df['批次号'] == batch]['30分钟溶解度']
    print(f"\n批次 {batch}:")
    print(f"  样本数: {len(batch_data)}")
    print(f"  均值: {batch_data.mean():.2f}%")
    print(f"  标准差: {batch_data.std():.2f}%")
    print(f"  最小值: {batch_data.min():.2f}%")
    print(f"  最大值: {batch_data.max():.2f}%")
    print(f"  中位数: {batch_data.median():.2f}%")
    print(f"  变异系数: {(batch_data.std()/batch_data.mean()*100):.2f}%")

# 检查USP711标准的关键阈值
print("\n4. USP711标准阈值分析:")
print("Q = 80% (标准值)")
print("Q+5% = 85% (第一阶段通过标准)")
print("Q-15% = 65% (第二、三阶段最低标准)")
print("Q-25% = 55% (第三阶段绝对最低标准)")

# 统计各批次在不同阈值下的药片数量
thresholds = [55, 65, 80, 85]
threshold_names = ['Q-25%', 'Q-15%', 'Q', 'Q+5%']

print("\n5. 各批次在不同阈值下的药片分布:")
for batch in ['B001', 'B002', 'B003']:
    batch_data = df[df['批次号'] == batch]['30分钟溶解度']
    print(f"\n批次 {batch}:")
    for i, threshold in enumerate(thresholds):
        count_above = (batch_data >= threshold).sum()
        count_below = (batch_data < threshold).sum()
        print(f"  ≥{threshold}% ({threshold_names[i]}): {count_above}片 ({count_above/24*100:.1f}%)")
        print(f"  <{threshold}% ({threshold_names[i]}): {count_below}片 ({count_below/24*100:.1f}%)")

# 创建可视化图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 各批次溶解度分布箱线图
df.boxplot(column='30分钟溶解度', by='批次号', ax=axes[0,0])
axes[0,0].set_title('各批次溶解度分布箱线图')
axes[0,0].set_xlabel('批次号')
axes[0,0].set_ylabel('30分钟溶解度 (%)')
axes[0,0].axhline(y=85, color='red', linestyle='--', alpha=0.7, label='Q+5% (85%)')
axes[0,0].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Q (80%)')
axes[0,0].axhline(y=65, color='blue', linestyle='--', alpha=0.7, label='Q-15% (65%)')
axes[0,0].axhline(y=55, color='purple', linestyle='--', alpha=0.7, label='Q-25% (55%)')
axes[0,0].legend()

# 2. 各批次溶解度分布直方图
for i, batch in enumerate(['B001', 'B002', 'B003']):
    batch_data = df[df['批次号'] == batch]['30分钟溶解度']
    axes[0,1].hist(batch_data, alpha=0.7, label=f'{batch}', bins=10)
axes[0,1].set_title('各批次溶解度分布直方图')
axes[0,1].set_xlabel('30分钟溶解度 (%)')
axes[0,1].set_ylabel('频数')
axes[0,1].axvline(x=85, color='red', linestyle='--', alpha=0.7)
axes[0,1].axvline(x=80, color='orange', linestyle='--', alpha=0.7)
axes[0,1].axvline(x=65, color='blue', linestyle='--', alpha=0.7)
axes[0,1].axvline(x=55, color='purple', linestyle='--', alpha=0.7)
axes[0,1].legend()

# 3. 各批次溶解度趋势图
for batch in ['B001', 'B002', 'B003']:
    batch_data = df[df['批次号'] == batch]['30分钟溶解度']
    axes[1,0].plot(range(1, 25), batch_data, marker='o', label=batch, alpha=0.7)
axes[1,0].set_title('各批次内药片溶解度变化趋势')
axes[1,0].set_xlabel('药片编号')
axes[1,0].set_ylabel('30分钟溶解度 (%)')
axes[1,0].axhline(y=85, color='red', linestyle='--', alpha=0.5)
axes[1,0].axhline(y=80, color='orange', linestyle='--', alpha=0.5)
axes[1,0].axhline(y=65, color='blue', linestyle='--', alpha=0.5)
axes[1,0].legend()

# 4. 各批次统计指标对比
batch_means = [df[df['批次号'] == batch]['30分钟溶解度'].mean() for batch in ['B001', 'B002', 'B003']]
batch_stds = [df[df['批次号'] == batch]['30分钟溶解度'].std() for batch in ['B001', 'B002', 'B003']]

x = np.arange(len(['B001', 'B002', 'B003']))
width = 0.35

axes[1,1].bar(x - width/2, batch_means, width, label='均值', alpha=0.8)
axes[1,1].bar(x + width/2, batch_stds, width, label='标准差', alpha=0.8)
axes[1,1].set_title('各批次统计指标对比')
axes[1,1].set_xlabel('批次号')
axes[1,1].set_ylabel('数值')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(['B001', 'B002', 'B003'])
axes[1,1].legend()

plt.tight_layout()
plt.savefig('f:\\25DIA\\code1.1\\描述性统计分析图.png', dpi=300, bbox_inches='tight')
plt.show()

# 进行正态性检验
print("\n6. 正态性检验 (Shapiro-Wilk检验):")
for batch in ['B001', 'B002', 'B003']:
    batch_data = df[df['批次号'] == batch]['30分钟溶解度']
    stat, p_value = stats.shapiro(batch_data)
    print(f"批次 {batch}: 统计量={stat:.4f}, p值={p_value:.4f}")
    if p_value > 0.05:
        print(f"  结论: 数据符合正态分布 (p > 0.05)")
    else:
        print(f"  结论: 数据不符合正态分布 (p ≤ 0.05)")

# 方差齐性检验
print("\n7. 方差齐性检验 (Levene检验):")
b001_data = df[df['批次号'] == 'B001']['30分钟溶解度']
b002_data = df[df['批次号'] == 'B002']['30分钟溶解度']
b003_data = df[df['批次号'] == 'B003']['30分钟溶解度']

stat, p_value = stats.levene(b001_data, b002_data, b003_data)
print(f"Levene统计量: {stat:.4f}, p值: {p_value:.4f}")
if p_value > 0.05:
    print("结论: 各批次方差齐性 (p > 0.05)")
else:
    print("结论: 各批次方差不齐 (p ≤ 0.05)")

# 单因素方差分析
print("\n8. 单因素方差分析 (ANOVA):")
stat, p_value = stats.f_oneway(b001_data, b002_data, b003_data)
print(f"F统计量: {stat:.4f}, p值: {p_value:.4f}")
if p_value > 0.05:
    print("结论: 各批次均值无显著差异 (p > 0.05)")
else:
    print("结论: 各批次均值存在显著差异 (p ≤ 0.05)")

print("\n=== 数据预处理完成 ===")