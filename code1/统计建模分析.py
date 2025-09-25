"""
基于现有批次数据建立统计模型
用于预测未来批次的溶解度测试失败率
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class StatisticalModel:
    def __init__(self, data_file='溶解度数据.csv'):
        """初始化统计模型"""
        self.data = pd.read_csv(data_file)
        self.batch_stats = {}
        self.population_params = {}
        self.models = {}
        
    def analyze_batch_characteristics(self):
        """分析各批次特征"""
        print("=== 批次特征分析 ===")
        
        batch_summary = []
        
        for batch in self.data['batch_id'].unique():
            batch_data = self.data[self.data['batch_id'] == batch]['dissolution'].values
            
            # 基本统计量
            stats_dict = {
                'batch': batch,
                'n': len(batch_data),
                'mean': np.mean(batch_data),
                'std': np.std(batch_data, ddof=1),
                'min': np.min(batch_data),
                'max': np.max(batch_data),
                'q25': np.percentile(batch_data, 25),
                'median': np.median(batch_data),
                'q75': np.percentile(batch_data, 75),
                'cv': np.std(batch_data, ddof=1) / np.mean(batch_data),  # 变异系数
                'skewness': stats.skew(batch_data),
                'kurtosis': stats.kurtosis(batch_data)
            }
            
            # 正态性检验
            shapiro_stat, shapiro_p = stats.shapiro(batch_data)
            stats_dict['shapiro_p'] = shapiro_p
            stats_dict['is_normal'] = shapiro_p > 0.05
            
            # Anderson-Darling检验
            ad_stat, ad_critical, ad_significance = stats.anderson(batch_data, dist='norm')
            stats_dict['anderson_stat'] = ad_stat
            
            # Kolmogorov-Smirnov检验
            ks_stat, ks_p = stats.kstest(batch_data, 'norm', 
                                        args=(np.mean(batch_data), np.std(batch_data, ddof=1)))
            stats_dict['ks_p'] = ks_p
            
            self.batch_stats[batch] = stats_dict
            batch_summary.append(stats_dict)
            
            print(f"批次 {batch}:")
            print(f"  样本数: {stats_dict['n']}")
            print(f"  均值±标准差: {stats_dict['mean']:.2f}±{stats_dict['std']:.2f}%")
            print(f"  范围: [{stats_dict['min']:.1f}, {stats_dict['max']:.1f}]%")
            print(f"  变异系数: {stats_dict['cv']:.4f}")
            print(f"  偏度: {stats_dict['skewness']:.3f}")
            print(f"  峰度: {stats_dict['kurtosis']:.3f}")
            print(f"  正态性检验 (Shapiro): p={stats_dict['shapiro_p']:.4f}")
            print(f"  正态性检验 (K-S): p={stats_dict['ks_p']:.4f}")
            print()
        
        # 保存批次统计摘要
        summary_df = pd.DataFrame(batch_summary)
        summary_df.to_csv('批次详细统计.csv', index=False)
        print("详细统计信息已保存到 '批次详细统计.csv'")
        
        return summary_df
    
    def estimate_population_parameters(self):
        """估计总体分布参数"""
        print("\n=== 总体分布参数估计 ===")
        
        all_data = self.data['dissolution'].values
        
        # 1. 正态分布参数估计
        mu_mle, sigma_mle = stats.norm.fit(all_data)
        
        # 2. 矩估计
        mu_mom = np.mean(all_data)
        sigma_mom = np.std(all_data, ddof=1)
        
        # 3. 贝叶斯估计（假设先验分布）
        # 假设均值的先验为正态分布，方差的先验为逆伽马分布
        n = len(all_data)
        sample_mean = np.mean(all_data)
        sample_var = np.var(all_data, ddof=1)
        
        # 贝叶斯后验参数（共轭先验）
        # 假设先验参数
        mu0 = 80  # 先验均值
        kappa0 = 1  # 先验精度参数
        alpha0 = 1  # 先验形状参数
        beta0 = 1   # 先验尺度参数
        
        # 后验参数
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * sample_mean) / kappa_n
        alpha_n = alpha0 + n / 2
        beta_n = beta0 + 0.5 * np.sum((all_data - sample_mean)**2) + \
                 (kappa0 * n * (sample_mean - mu0)**2) / (2 * kappa_n)
        
        # 后验均值估计
        mu_bayes = mu_n
        sigma2_bayes = beta_n / (alpha_n - 1)
        sigma_bayes = np.sqrt(sigma2_bayes)
        
        self.population_params = {
            'mle': {'mu': mu_mle, 'sigma': sigma_mle},
            'mom': {'mu': mu_mom, 'sigma': sigma_mom},
            'bayes': {'mu': mu_bayes, 'sigma': sigma_bayes},
            'sample_size': n
        }
        
        print("最大似然估计 (MLE):")
        print(f"  μ = {mu_mle:.3f}%, σ = {sigma_mle:.3f}%")
        
        print("矩估计 (MOM):")
        print(f"  μ = {mu_mom:.3f}%, σ = {sigma_mom:.3f}%")
        
        print("贝叶斯估计:")
        print(f"  μ = {mu_bayes:.3f}%, σ = {sigma_bayes:.3f}%")
        
        # 4. 分布拟合优度检验
        self._test_distribution_fit(all_data)
        
        return self.population_params
    
    def _test_distribution_fit(self, data):
        """测试不同分布的拟合优度"""
        print("\n--- 分布拟合优度检验 ---")
        
        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'gamma': stats.gamma,
            'beta': stats.beta,
            'weibull': stats.weibull_min
        }
        
        fit_results = {}
        
        for dist_name, dist in distributions.items():
            try:
                # 参数估计
                if dist_name == 'beta':
                    # Beta分布需要数据在[0,1]范围内
                    scaled_data = (data - data.min()) / (data.max() - data.min())
                    params = dist.fit(scaled_data)
                    # K-S检验
                    ks_stat, ks_p = stats.kstest(scaled_data, 
                                                lambda x: dist.cdf(x, *params))
                else:
                    params = dist.fit(data)
                    # K-S检验
                    ks_stat, ks_p = stats.kstest(data, 
                                                lambda x: dist.cdf(x, *params))
                
                # AIC计算
                log_likelihood = np.sum(dist.logpdf(data if dist_name != 'beta' else scaled_data, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                
                fit_results[dist_name] = {
                    'params': params,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'aic': aic,
                    'log_likelihood': log_likelihood
                }
                
                print(f"{dist_name.capitalize()}分布:")
                print(f"  K-S检验: 统计量={ks_stat:.4f}, p值={ks_p:.4f}")
                print(f"  AIC: {aic:.2f}")
                
            except Exception as e:
                print(f"{dist_name}分布拟合失败: {e}")
        
        # 选择最佳分布（基于AIC）
        if fit_results:
            best_dist = min(fit_results.keys(), key=lambda x: fit_results[x]['aic'])
            print(f"\n最佳拟合分布: {best_dist} (最小AIC: {fit_results[best_dist]['aic']:.2f})")
            self.best_distribution = best_dist
            self.fit_results = fit_results
    
    def hierarchical_modeling(self):
        """层次模型：考虑批次间和批次内变异"""
        print("\n=== 层次模型分析 ===")
        
        # 计算批次间和批次内变异
        batch_means = []
        batch_vars = []
        
        for batch in self.data['batch_id'].unique():
            batch_data = self.data[self.data['batch_id'] == batch]['dissolution'].values
            batch_means.append(np.mean(batch_data))
            batch_vars.append(np.var(batch_data, ddof=1))
        
        batch_means = np.array(batch_means)
        batch_vars = np.array(batch_vars)
        
        # 总体均值
        grand_mean = np.mean(batch_means)
        
        # 批次间变异
        between_batch_var = np.var(batch_means, ddof=1)
        
        # 批次内变异（平均）
        within_batch_var = np.mean(batch_vars)
        
        # 总变异
        total_var = between_batch_var + within_batch_var
        
        # 组内相关系数 (ICC)
        icc = between_batch_var / total_var
        
        print(f"总体均值: {grand_mean:.3f}%")
        print(f"批次间变异: {between_batch_var:.3f}")
        print(f"批次内变异: {within_batch_var:.3f}")
        print(f"总变异: {total_var:.3f}")
        print(f"组内相关系数 (ICC): {icc:.3f}")
        
        # 方差分量分析
        self._variance_components_analysis()
        
        self.hierarchical_params = {
            'grand_mean': grand_mean,
            'between_batch_var': between_batch_var,
            'within_batch_var': within_batch_var,
            'total_var': total_var,
            'icc': icc
        }
        
        return self.hierarchical_params
    
    def _variance_components_analysis(self):
        """方差分量分析（单因素随机效应模型）"""
        print("\n--- 方差分量分析 ---")
        
        # 准备数据进行ANOVA
        from scipy.stats import f_oneway
        
        batch_groups = []
        for batch in self.data['batch_id'].unique():
            batch_data = self.data[self.data['batch_id'] == batch]['dissolution'].values
            batch_groups.append(batch_data)
        
        # 单因素ANOVA
        f_stat, p_value = f_oneway(*batch_groups)
        
        print(f"单因素ANOVA:")
        print(f"  F统计量: {f_stat:.4f}")
        print(f"  p值: {p_value:.4f}")
        print(f"  批次间差异显著性: {'显著' if p_value < 0.05 else '不显著'}")
        
        # 计算均方
        k = len(batch_groups)  # 组数
        n_total = sum(len(group) for group in batch_groups)
        n_per_group = n_total / k  # 假设平衡设计
        
        # 组间均方和组内均方
        ss_between = sum(len(group) * (np.mean(group) - np.mean(self.data['dissolution']))**2 
                        for group in batch_groups)
        ms_between = ss_between / (k - 1)
        
        ss_within = sum(sum((x - np.mean(group))**2 for x in group) for group in batch_groups)
        ms_within = ss_within / (n_total - k)
        
        # 方差分量估计
        sigma2_within = ms_within
        sigma2_between = max(0, (ms_between - ms_within) / n_per_group)
        
        print(f"  组内方差分量 (σ²_within): {sigma2_within:.3f}")
        print(f"  组间方差分量 (σ²_between): {sigma2_between:.3f}")
    
    def predictive_modeling(self):
        """建立预测模型"""
        print("\n=== 预测模型建立 ===")
        
        # 1. 基于历史数据的贝叶斯预测
        self._bayesian_prediction()
        
        # 2. 混合高斯模型
        self._gaussian_mixture_model()
        
        # 3. 基于置信区间的预测
        self._confidence_interval_prediction()
    
    def _bayesian_prediction(self):
        """贝叶斯预测"""
        print("\n--- 贝叶斯预测模型 ---")
        
        # 使用之前估计的贝叶斯参数
        bayes_params = self.population_params['bayes']
        mu = bayes_params['mu']
        sigma = bayes_params['sigma']
        
        # 预测新批次的分布
        print(f"预测新批次溶解度分布: N({mu:.2f}, {sigma:.2f}²)")
        
        # 计算各种概率
        prob_below_75 = stats.norm.cdf(75, mu, sigma)
        prob_below_80 = stats.norm.cdf(80, mu, sigma)
        prob_above_85 = 1 - stats.norm.cdf(85, mu, sigma)
        
        print(f"预测概率:")
        print(f"  P(溶解度 < 75%) = {prob_below_75:.4f}")
        print(f"  P(溶解度 < 80%) = {prob_below_80:.4f}")
        print(f"  P(溶解度 > 85%) = {prob_above_85:.4f}")
        
        # 预测区间
        ci_95 = stats.norm.interval(0.95, mu, sigma)
        ci_99 = stats.norm.interval(0.99, mu, sigma)
        
        print(f"预测区间:")
        print(f"  95%预测区间: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]%")
        print(f"  99%预测区间: [{ci_99[0]:.2f}, {ci_99[1]:.2f}]%")
    
    def _gaussian_mixture_model(self):
        """高斯混合模型"""
        print("\n--- 高斯混合模型 ---")
        
        data = self.data['dissolution'].values.reshape(-1, 1)
        
        # 尝试不同的组件数
        n_components_range = range(1, 5)
        bic_scores = []
        aic_scores = []
        
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(data)
            bic_scores.append(gmm.bic(data))
            aic_scores.append(gmm.aic(data))
        
        # 选择最佳组件数
        best_n_components = n_components_range[np.argmin(bic_scores)]
        
        print(f"最佳组件数: {best_n_components} (基于BIC)")
        
        # 拟合最佳模型
        best_gmm = GaussianMixture(n_components=best_n_components, random_state=42)
        best_gmm.fit(data)
        
        print(f"混合权重: {best_gmm.weights_}")
        print(f"均值: {best_gmm.means_.flatten()}")
        print(f"协方差: {best_gmm.covariances_.flatten()}")
        
        self.models['gmm'] = best_gmm
    
    def _confidence_interval_prediction(self):
        """基于置信区间的预测"""
        print("\n--- 置信区间预测 ---")
        
        all_data = self.data['dissolution'].values
        n = len(all_data)
        mean = np.mean(all_data)
        std = np.std(all_data, ddof=1)
        se = std / np.sqrt(n)
        
        # t分布置信区间
        alpha = 0.05
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
        ci_lower = mean - t_critical * se
        ci_upper = mean + t_critical * se
        
        print(f"总体均值的95%置信区间: [{ci_lower:.2f}, {ci_upper:.2f}]%")
        
        # 预测区间（考虑新观测的不确定性）
        pred_se = std * np.sqrt(1 + 1/n)
        pred_lower = mean - t_critical * pred_se
        pred_upper = mean + t_critical * pred_se
        
        print(f"新观测值的95%预测区间: [{pred_lower:.2f}, {pred_upper:.2f}]%")
    
    def model_validation(self):
        """模型验证"""
        print("\n=== 模型验证 ===")
        
        # 留一法交叉验证
        self._cross_validation()
        
        # 残差分析
        self._residual_analysis()
    
    def _cross_validation(self):
        """交叉验证"""
        print("\n--- 交叉验证 ---")
        
        # 对每个批次进行留一批次验证
        batches = self.data['batch_id'].unique()
        predictions = []
        actuals = []
        
        for test_batch in batches:
            # 训练数据（除了测试批次）
            train_data = self.data[self.data['batch_id'] != test_batch]['dissolution'].values
            test_data = self.data[self.data['batch_id'] == test_batch]['dissolution'].values
            
            # 用训练数据估计参数
            train_mean = np.mean(train_data)
            train_std = np.std(train_data, ddof=1)
            
            # 预测测试批次的均值
            predicted_mean = train_mean
            actual_mean = np.mean(test_data)
            
            predictions.append(predicted_mean)
            actuals.append(actual_mean)
            
            print(f"批次 {test_batch}: 预测={predicted_mean:.2f}%, 实际={actual_mean:.2f}%")
        
        # 计算预测误差
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
        
        print(f"\n交叉验证结果:")
        print(f"  均方误差 (MSE): {mse:.4f}")
        print(f"  均方根误差 (RMSE): {rmse:.4f}")
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
    
    def _residual_analysis(self):
        """残差分析"""
        print("\n--- 残差分析 ---")
        
        # 计算标准化残差
        all_data = self.data['dissolution'].values
        mean = np.mean(all_data)
        std = np.std(all_data, ddof=1)
        
        residuals = (all_data - mean) / std
        
        # 正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        print(f"残差正态性检验 (Shapiro-Wilk):")
        print(f"  统计量: {shapiro_stat:.4f}")
        print(f"  p值: {shapiro_p:.4f}")
        print(f"  结论: {'残差服从正态分布' if shapiro_p > 0.05 else '残差不服从正态分布'}")
        
        # 异常值检测
        outliers = np.abs(residuals) > 2.5
        n_outliers = np.sum(outliers)
        
        print(f"\n异常值检测 (|标准化残差| > 2.5):")
        print(f"  异常值数量: {n_outliers}")
        print(f"  异常值比例: {n_outliers/len(residuals)*100:.2f}%")
        
        if n_outliers > 0:
            outlier_indices = np.where(outliers)[0]
            print(f"  异常值位置: {outlier_indices}")
    
    def create_model_visualizations(self):
        """创建模型可视化"""
        print("\n=== 生成模型可视化 ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('统计模型分析结果', fontsize=16, fontweight='bold')
        
        # 1. 批次分布对比
        for i, batch in enumerate(self.data['batch_id'].unique()):
            batch_data = self.data[self.data['batch_id'] == batch]['dissolution'].values
            axes[0,0].hist(batch_data, alpha=0.7, label=f'批次{batch}', bins=10)
        
        axes[0,0].set_xlabel('溶解度 (%)')
        axes[0,0].set_ylabel('频数')
        axes[0,0].set_title('各批次溶解度分布')
        axes[0,0].legend()
        
        # 2. Q-Q图
        all_data = self.data['dissolution'].values
        stats.probplot(all_data, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('正态Q-Q图')
        
        # 3. 批次均值和变异
        batch_means = [self.batch_stats[batch]['mean'] for batch in self.batch_stats.keys()]
        batch_stds = [self.batch_stats[batch]['std'] for batch in self.batch_stats.keys()]
        batch_names = list(self.batch_stats.keys())
        
        x_pos = np.arange(len(batch_names))
        axes[0,2].bar(x_pos, batch_means, yerr=batch_stds, capsize=5, alpha=0.7)
        axes[0,2].set_xlabel('批次')
        axes[0,2].set_ylabel('溶解度 (%)')
        axes[0,2].set_title('批次均值±标准差')
        axes[0,2].set_xticks(x_pos)
        axes[0,2].set_xticklabels(batch_names)
        
        # 4. 残差图
        mean = np.mean(all_data)
        residuals = all_data - mean
        axes[1,0].scatter(range(len(residuals)), residuals, alpha=0.6)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('观测序号')
        axes[1,0].set_ylabel('残差')
        axes[1,0].set_title('残差图')
        
        # 5. 分布拟合对比
        x = np.linspace(all_data.min(), all_data.max(), 100)
        axes[1,1].hist(all_data, bins=15, density=True, alpha=0.7, label='观测数据')
        
        # 正态分布拟合
        mu, sigma = stats.norm.fit(all_data)
        axes[1,1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label=f'正态分布 N({mu:.1f},{sigma:.1f})')
        
        axes[1,1].set_xlabel('溶解度 (%)')
        axes[1,1].set_ylabel('密度')
        axes[1,1].set_title('分布拟合')
        axes[1,1].legend()
        
        # 6. 预测区间可视化
        ci_95 = stats.norm.interval(0.95, mu, sigma)
        ci_99 = stats.norm.interval(0.99, mu, sigma)
        
        axes[1,2].axvspan(ci_95[0], ci_95[1], alpha=0.3, color='blue', label='95%预测区间')
        axes[1,2].axvspan(ci_99[0], ci_99[1], alpha=0.2, color='red', label='99%预测区间')
        axes[1,2].axvline(mu, color='black', linestyle='-', label=f'预测均值 ({mu:.1f}%)')
        axes[1,2].hist(all_data, bins=15, density=True, alpha=0.5, color='gray')
        axes[1,2].set_xlabel('溶解度 (%)')
        axes[1,2].set_ylabel('密度')
        axes[1,2].set_title('预测区间')
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig('统计模型分析图.png', dpi=300, bbox_inches='tight')
        print("模型分析图表已保存为 '统计模型分析图.png'")
        plt.show()
        
        return fig

def main():
    """主函数"""
    print("USP711溶解度测试统计建模分析")
    print("=" * 50)
    
    # 初始化模型
    model = StatisticalModel()
    
    # 1. 批次特征分析
    batch_summary = model.analyze_batch_characteristics()
    
    # 2. 总体参数估计
    population_params = model.estimate_population_parameters()
    
    # 3. 层次模型分析
    hierarchical_params = model.hierarchical_modeling()
    
    # 4. 预测建模
    model.predictive_modeling()
    
    # 5. 模型验证
    model.model_validation()
    
    # 6. 可视化
    model.create_model_visualizations()
    
    # 保存模型结果
    model_results = {
        'population_parameters': population_params,
        'hierarchical_parameters': hierarchical_params,
        'batch_statistics': model.batch_stats
    }
    
    # 保存到文件
    import json
    with open('statistical_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(model_results, f, ensure_ascii=False, indent=2)
    
    print("\n统计模型结果已保存到 'statistical_model_results.json'")
    
    return model_results

if __name__ == "__main__":
    results = main()