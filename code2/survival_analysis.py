#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复赛大题（二）生存分析计算
临床试验设计中的检验效能计算
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from math import log, exp, sqrt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SurvivalTrialAnalysis:
    """生存分析临床试验设计类"""
    
    def __init__(self):
        self.alpha = 0.025  # 单侧显著性水平
        
    def exponential_hazard_from_survival(self, survival_rate, time_years):
        """从生存率计算指数分布的风险率"""
        return -log(survival_rate) / time_years
    
    def calculate_events_schoenfeld(self, n, hazard_ratio, alpha=0.025, power=0.8):
        """使用Schoenfeld公式计算所需事件数"""
        z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)
        
        # Schoenfeld公式
        events = 4 * (z_alpha + z_beta)**2 / (log(hazard_ratio))**2
        return events
    
    def calculate_power_logrank(self, n_events, hazard_ratio, alpha=0.025):
        """计算log-rank检验的检验效能"""
        z_alpha = norm.ppf(1 - alpha)
        
        # 检验统计量的非中心参数
        delta = sqrt(n_events / 4) * abs(log(hazard_ratio))
        
        # 计算检验效能
        power = 1 - norm.cdf(z_alpha - delta)
        return power
    
    def simulate_exponential_survival(self, n, hazard_rate, follow_up_time):
        """模拟指数分布生存时间"""
        # 生成指数分布的生存时间
        survival_times = np.random.exponential(1/hazard_rate, n)
        
        # 考虑随访截止时间
        observed_times = np.minimum(survival_times, follow_up_time)
        events = survival_times <= follow_up_time
        
        return observed_times, events
    
    def calculate_expected_events(self, n, hazard_rate, accrual_rate, follow_up_time):
        """计算期望事件数（考虑入组时间）"""
        accrual_time = n / accrual_rate
        
        # 对于指数分布，考虑入组时间的期望事件数
        if accrual_time >= follow_up_time:
            # 入组时间超过随访时间
            expected_events = n * (1 - exp(-hazard_rate * follow_up_time))
        else:
            # 正常情况
            term1 = n * (1 - exp(-hazard_rate * follow_up_time))
            term2 = (hazard_rate * n / accrual_rate) * (follow_up_time - accrual_time + 
                    (1 - exp(-hazard_rate * accrual_time)) / hazard_rate)
            expected_events = term1 - term2
            
        return max(0, expected_events)
    
    def problem_2_1_1(self):
        """问题2.1.1：计算阳性人群和全人群的检验效能"""
        print("=" * 60)
        print("问题2.1.1：检验效能计算")
        print("=" * 60)
        
        # 基本参数
        n_positive = 160  # 阳性人群样本量
        accrual_rate = 10  # 入组速率（人/月）
        follow_up_years = 3  # 随访时间（年）
        
        # 阳性人群参数
        survival_2y_positive_control = 0.5  # 对照组2年生存率
        hr_positive = 0.51  # 风险比
        
        # 阴性人群参数
        survival_2y_negative_control = 0.7  # 对照组2年生存率
        hr_negative_options = [0.57, 0.67, 0.76]  # 三种风险比选项
        
        # 计算风险率
        hazard_positive_control = self.exponential_hazard_from_survival(survival_2y_positive_control, 2)
        hazard_positive_treatment = hazard_positive_control * hr_positive
        
        hazard_negative_control = self.exponential_hazard_from_survival(survival_2y_negative_control, 2)
        
        print(f"阳性人群对照组风险率: {hazard_positive_control:.4f}")
        print(f"阳性人群试验组风险率: {hazard_positive_treatment:.4f}")
        print(f"阴性人群对照组风险率: {hazard_negative_control:.4f}")
        print()
        
        # 计算阳性人群期望事件数
        events_positive_control = self.calculate_expected_events(
            n_positive//2, hazard_positive_control, accrual_rate//2, follow_up_years
        )
        events_positive_treatment = self.calculate_expected_events(
            n_positive//2, hazard_positive_treatment, accrual_rate//2, follow_up_years
        )
        total_events_positive = events_positive_control + events_positive_treatment
        
        print(f"阳性人群期望事件数: {total_events_positive:.1f}")
        
        # 计算阳性人群检验效能
        power_positive = self.calculate_power_logrank(total_events_positive, hr_positive)
        print(f"阳性人群检验效能: {power_positive:.4f}")
        print()
        
        # 计算全人群检验效能（三种HR情况）
        results = []
        for hr_negative in hr_negative_options:
            hazard_negative_treatment = hazard_negative_control * hr_negative
            
            # 阴性人群期望事件数
            events_negative_control = self.calculate_expected_events(
                n_positive//2, hazard_negative_control, accrual_rate//2, follow_up_years
            )
            events_negative_treatment = self.calculate_expected_events(
                n_positive//2, hazard_negative_treatment, accrual_rate//2, follow_up_years
            )
            total_events_negative = events_negative_control + events_negative_treatment
            
            # 全人群总事件数
            total_events_overall = total_events_positive + total_events_negative
            
            # 全人群加权HR
            weight_positive = total_events_positive / total_events_overall
            weight_negative = total_events_negative / total_events_overall
            hr_overall = exp(weight_positive * log(hr_positive) + weight_negative * log(hr_negative))
            
            # 全人群检验效能
            power_overall = self.calculate_power_logrank(total_events_overall, hr_overall)
            
            results.append({
                'HR_negative': hr_negative,
                'Events_negative': total_events_negative,
                'Events_overall': total_events_overall,
                'HR_overall': hr_overall,
                'Power_overall': power_overall
            })
            
            print(f"阴性人群HR={hr_negative}:")
            print(f"  阴性人群期望事件数: {total_events_negative:.1f}")
            print(f"  全人群期望事件数: {total_events_overall:.1f}")
            print(f"  全人群加权HR: {hr_overall:.4f}")
            print(f"  全人群检验效能: {power_overall:.4f}")
            print()
        
        return {
            'power_positive': power_positive,
            'events_positive': total_events_positive,
            'results_overall': results
        }
    
    def problem_2_1_2(self):
        """问题2.1.2：研究阴性人群比例对检验效能的影响"""
        print("=" * 60)
        print("问题2.1.2：阴性人群比例对检验效能的影响")
        print("=" * 60)
        
        # 基本参数
        n_positive = 160
        accrual_rate = 10
        follow_up_years = 3
        
        # 阳性人群参数（固定）
        survival_2y_positive_control = 0.5
        hr_positive = 0.51
        hazard_positive_control = self.exponential_hazard_from_survival(survival_2y_positive_control, 2)
        hazard_positive_treatment = hazard_positive_control * hr_positive
        
        # 阴性人群参数
        survival_2y_negative_control = 0.7
        hazard_negative_control = self.exponential_hazard_from_survival(survival_2y_negative_control, 2)
        
        # 阳性人群基准检验效能
        events_positive_control = self.calculate_expected_events(
            n_positive//2, hazard_positive_control, accrual_rate//2, follow_up_years
        )
        events_positive_treatment = self.calculate_expected_events(
            n_positive//2, hazard_positive_treatment, accrual_rate//2, follow_up_years
        )
        total_events_positive = events_positive_control + events_positive_treatment
        power_baseline = self.calculate_power_logrank(total_events_positive, hr_positive)
        
        print(f"基准检验效能（仅阳性人群）: {power_baseline:.4f}")
        print()
        
        # 阴性人群比例范围
        negative_proportions = np.arange(0, 0.71, 0.05)
        hr_negative_range = np.arange(0.3, 1.1, 0.05)
        
        results = []
        
        for neg_prop in negative_proportions:
            if neg_prop == 0:
                continue
                
            # 计算样本量分配
            total_n = n_positive / (1 - neg_prop)
            n_negative = int(total_n * neg_prop)
            
            for hr_negative in hr_negative_range:
                hazard_negative_treatment = hazard_negative_control * hr_negative
                
                # 阴性人群期望事件数
                events_negative_control = self.calculate_expected_events(
                    n_negative//2, hazard_negative_control, accrual_rate//2, follow_up_years
                )
                events_negative_treatment = self.calculate_expected_events(
                    n_negative//2, hazard_negative_treatment, accrual_rate//2, follow_up_years
                )
                total_events_negative = events_negative_control + events_negative_treatment
                
                # 全人群总事件数和加权HR
                total_events_overall = total_events_positive + total_events_negative
                weight_positive = total_events_positive / total_events_overall
                weight_negative = total_events_negative / total_events_overall
                hr_overall = exp(weight_positive * log(hr_positive) + weight_negative * log(hr_negative))
                
                # 全人群检验效能
                power_overall = self.calculate_power_logrank(total_events_overall, hr_overall)
                
                results.append({
                    'negative_proportion': neg_prop,
                    'hr_negative': hr_negative,
                    'power_overall': power_overall,
                    'power_improvement': power_overall - power_baseline
                })
        
        # 找到对检验效能有正向影响的HR阈值
        df_results = pd.DataFrame(results)
        
        print("阴性人群HR对全人群检验效能正向影响的阈值分析：")
        for neg_prop in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            subset = df_results[df_results['negative_proportion'] == neg_prop]
            positive_impact = subset[subset['power_improvement'] > 0]
            if len(positive_impact) > 0:
                hr_threshold = positive_impact['hr_negative'].max()
                print(f"阴性人群比例{neg_prop:.0%}时，HR < {hr_threshold:.3f} 有正向影响")
        
        return df_results
    
    def problem_2_2(self):
        """问题2.2：图示法alpha分配优化"""
        print("=" * 60)
        print("问题2.2：图示法alpha分配优化")
        print("=" * 60)
        
        # 基本参数
        total_n = 400
        n_positive = n_negative = 200  # 各占50%
        accrual_rate_total = 20  # 总入组速率
        target_events = 300  # 目标事件数
        
        # 生存参数
        median_survival_control = 12  # 对照组中位生存期（月）
        hazard_control = log(2) / median_survival_control
        
        hr_positive = 0.6  # 阳性人群HR
        hr_negative = 0.8  # 阴性人群HR
        
        hazard_positive_treatment = hazard_control * hr_positive
        hazard_negative_treatment = hazard_control * hr_negative
        
        print(f"对照组风险率: {hazard_control:.4f} /月")
        print(f"阳性人群试验组风险率: {hazard_positive_treatment:.4f} /月")
        print(f"阴性人群试验组风险率: {hazard_negative_treatment:.4f} /月")
        print()
        
        # 计算期望事件数分布
        # 简化计算：假设事件数按人群比例分配
        events_positive = target_events * 0.5  # 假设各人群贡献相等的事件数
        events_negative = target_events * 0.5
        
        print(f"阳性人群期望事件数: {events_positive:.0f}")
        print(f"阴性人群期望事件数: {events_negative:.0f}")
        print()
        
        # Alpha分配优化
        alpha_total = 0.025
        alpha_range = np.arange(0.001, alpha_total, 0.001)
        
        results_2_2_1 = []  # 两个人群均拒绝零假设
        results_2_2_2 = []  # 至少一个人群拒绝零假设
        
        for alpha1 in alpha_range:
            alpha2 = alpha_total - alpha1
            if alpha2 <= 0:
                continue
            
            # 计算各人群的检验效能
            power_overall = self.calculate_power_logrank(target_events, 
                                                       exp(0.5*log(hr_positive) + 0.5*log(hr_negative)), 
                                                       alpha1)
            power_positive = self.calculate_power_logrank(events_positive, hr_positive, alpha2)
            
            # 图示法：如果第一个检验拒绝，第二个检验可以使用全部alpha
            power_positive_full = self.calculate_power_logrank(events_positive, hr_positive, alpha_total)
            power_overall_full = self.calculate_power_logrank(target_events, 
                                                            exp(0.5*log(hr_positive) + 0.5*log(hr_negative)), 
                                                            alpha_total)
            
            # 2.2.1: 两个人群均拒绝零假设的概率
            prob_both_reject = power_overall * power_positive
            
            # 2.2.2: 至少一个人群拒绝零假设的概率
            prob_at_least_one = (power_overall * power_positive_full + 
                                power_positive * power_overall_full - 
                                power_overall * power_positive)
            
            results_2_2_1.append({
                'alpha1': alpha1,
                'alpha2': alpha2,
                'power_overall': power_overall,
                'power_positive': power_positive,
                'prob_both_reject': prob_both_reject
            })
            
            results_2_2_2.append({
                'alpha1': alpha1,
                'alpha2': alpha2,
                'prob_at_least_one': prob_at_least_one
            })
        
        # 找到最优分配
        df_2_2_1 = pd.DataFrame(results_2_2_1)
        df_2_2_2 = pd.DataFrame(results_2_2_2)
        
        optimal_2_2_1 = df_2_2_1.loc[df_2_2_1['prob_both_reject'].idxmax()]
        optimal_2_2_2 = df_2_2_2.loc[df_2_2_2['prob_at_least_one'].idxmax()]
        
        print("问题2.2.1结果：")
        print(f"最优alpha分配: α1={optimal_2_2_1['alpha1']:.4f}, α2={optimal_2_2_1['alpha2']:.4f}")
        print(f"两个人群均拒绝零假设的最大概率: {optimal_2_2_1['prob_both_reject']:.4f}")
        print()
        
        print("问题2.2.2结果：")
        print(f"最优alpha分配: α1={optimal_2_2_2['alpha1']:.4f}, α2={optimal_2_2_2['alpha2']:.4f}")
        print(f"至少一个人群拒绝零假设的最大概率: {optimal_2_2_2['prob_at_least_one']:.4f}")
        
        return {
            'results_2_2_1': df_2_2_1,
            'results_2_2_2': df_2_2_2,
            'optimal_2_2_1': optimal_2_2_1,
            'optimal_2_2_2': optimal_2_2_2
        }

def main():
    """主函数"""
    print("复赛大题（二）：临床试验生存分析")
    print("=" * 60)
    
    # 创建分析实例
    analyzer = SurvivalTrialAnalysis()
    
    # 问题2.1.1
    results_2_1_1 = analyzer.problem_2_1_1()
    
    # 问题2.1.2
    results_2_1_2 = analyzer.problem_2_1_2()
    
    # 问题2.2
    results_2_2 = analyzer.problem_2_2()
    
    print("\n" + "=" * 60)
    print("分析完成！")
    
    return {
        'results_2_1_1': results_2_1_1,
        'results_2_1_2': results_2_1_2,
        'results_2_2': results_2_2
    }

if __name__ == "__main__":
    results = main()