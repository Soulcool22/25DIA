"""
USP711溶解度测试标准实现
根据美国药典USP711标准实现三阶段溶解度测试判断逻辑
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import random

class USP711Tester:
    """USP711溶解度测试类"""
    
    def __init__(self, Q=80):
        """
        初始化测试参数
        
        Args:
            Q (float): 规定时间内溶解的活性药物成分指定量，默认80%
        """
        self.Q = Q
        self.Q_plus_5 = Q + 5    # 85%
        self.Q_minus_15 = Q - 15  # 65%
        self.Q_minus_25 = Q - 25  # 55%
    
    def stage1_test(self, dissolution_values: List[float]) -> Tuple[bool, str]:
        """
        第一阶段测试：6个药片，每个都要大于85%
        
        Args:
            dissolution_values: 6个药片的溶解度值
            
        Returns:
            (是否通过, 详细信息)
        """
        if len(dissolution_values) != 6:
            return False, f"第一阶段需要6个药片，实际提供{len(dissolution_values)}个"
        
        all_above_85 = all(val > self.Q_plus_5 for val in dissolution_values)
        
        if all_above_85:
            return True, f"第一阶段通过：所有6个药片溶解度都大于{self.Q_plus_5}%"
        else:
            below_85_count = sum(1 for val in dissolution_values if val <= self.Q_plus_5)
            return False, f"第一阶段未通过：有{below_85_count}个药片溶解度不大于{self.Q_plus_5}%"
    
    def stage2_test(self, dissolution_values: List[float]) -> Tuple[bool, str]:
        """
        第二阶段测试：12个药片，平均值>80%且没有<65%的药片
        
        Args:
            dissolution_values: 12个药片的溶解度值
            
        Returns:
            (是否通过, 详细信息)
        """
        if len(dissolution_values) != 12:
            return False, f"第二阶段需要12个药片，实际提供{len(dissolution_values)}个"
        
        mean_dissolution = np.mean(dissolution_values)
        below_65_count = sum(1 for val in dissolution_values if val < self.Q_minus_15)
        
        if mean_dissolution > self.Q and below_65_count == 0:
            return True, f"第二阶段通过：平均溶解度{mean_dissolution:.2f}%>{self.Q}%，且无药片<{self.Q_minus_15}%"
        else:
            reasons = []
            if mean_dissolution <= self.Q:
                reasons.append(f"平均溶解度{mean_dissolution:.2f}%不大于{self.Q}%")
            if below_65_count > 0:
                reasons.append(f"有{below_65_count}个药片溶解度<{self.Q_minus_15}%")
            return False, f"第二阶段未通过：{'; '.join(reasons)}"
    
    def stage3_test(self, dissolution_values: List[float]) -> Tuple[bool, str]:
        """
        第三阶段测试：24个药片，平均值>80%且<65%的不超过2片且无<55%的药片
        
        Args:
            dissolution_values: 24个药片的溶解度值
            
        Returns:
            (是否通过, 详细信息)
        """
        if len(dissolution_values) != 24:
            return False, f"第三阶段需要24个药片，实际提供{len(dissolution_values)}个"
        
        mean_dissolution = np.mean(dissolution_values)
        below_65_count = sum(1 for val in dissolution_values if val < self.Q_minus_15)
        below_55_count = sum(1 for val in dissolution_values if val < self.Q_minus_25)
        
        conditions_met = (
            mean_dissolution > self.Q and 
            below_65_count <= 2 and 
            below_55_count == 0
        )
        
        if conditions_met:
            return True, f"第三阶段通过：平均溶解度{mean_dissolution:.2f}%>{self.Q}%，{below_65_count}个药片<{self.Q_minus_15}%（≤2），{below_55_count}个药片<{self.Q_minus_25}%"
        else:
            reasons = []
            if mean_dissolution <= self.Q:
                reasons.append(f"平均溶解度{mean_dissolution:.2f}%不大于{self.Q}%")
            if below_65_count > 2:
                reasons.append(f"有{below_65_count}个药片溶解度<{self.Q_minus_15}%（>2）")
            if below_55_count > 0:
                reasons.append(f"有{below_55_count}个药片溶解度<{self.Q_minus_25}%")
            return False, f"第三阶段未通过：{'; '.join(reasons)}"
    
    def full_test(self, batch_dissolution_values: List[float]) -> Tuple[str, int, str]:
        """
        完整的三阶段测试流程
        
        Args:
            batch_dissolution_values: 批次中所有药片的溶解度值
            
        Returns:
            (测试结果, 失败阶段, 详细信息)
        """
        if len(batch_dissolution_values) < 24:
            return "数据不足", 0, f"需要至少24个药片数据，实际只有{len(batch_dissolution_values)}个"
        
        # 随机选择药片进行测试（模拟实际抽样过程）
        available_tablets = batch_dissolution_values.copy()
        random.shuffle(available_tablets)
        
        # 第一阶段：前6个药片
        stage1_tablets = available_tablets[:6]
        stage1_pass, stage1_info = self.stage1_test(stage1_tablets)
        
        if stage1_pass:
            return "通过", 0, f"第一阶段通过。{stage1_info}"
        
        # 第二阶段：前12个药片
        stage2_tablets = available_tablets[:12]
        stage2_pass, stage2_info = self.stage2_test(stage2_tablets)
        
        if stage2_pass:
            return "通过", 0, f"第二阶段通过。{stage2_info}"
        
        # 第三阶段：全部24个药片
        stage3_tablets = available_tablets[:24]
        stage3_pass, stage3_info = self.stage3_test(stage3_tablets)
        
        if stage3_pass:
            return "通过", 0, f"第三阶段通过。{stage3_info}"
        else:
            return "失败", 3, f"第三阶段失败。{stage3_info}"

def test_existing_batches():
    """测试现有的三个批次数据"""
    # 加载数据
    df = pd.read_csv('溶解度数据.csv')
    tester = USP711Tester()
    
    print("=== USP711溶解度测试结果 ===\n")
    
    for batch_id in df['batch_id'].unique():
        batch_data = df[df['batch_id'] == batch_id]['dissolution'].tolist()
        
        print(f"批次 {batch_id}:")
        print(f"  溶解度数据: {batch_data}")
        
        result, fail_stage, info = tester.full_test(batch_data)
        print(f"  测试结果: {result}")
        print(f"  详细信息: {info}")
        
        if result == "失败":
            print(f"  失败阶段: 第{fail_stage}阶段")
        
        print()

def analyze_stage_failure_patterns():
    """分析各阶段失败模式"""
    df = pd.read_csv('溶解度数据.csv')
    tester = USP711Tester()
    
    print("=== 各阶段失败模式分析 ===\n")
    
    for batch_id in df['batch_id'].unique():
        batch_data = df[df['batch_id'] == batch_id]['dissolution'].tolist()
        
        print(f"批次 {batch_id} 详细分析:")
        
        # 模拟第一阶段测试
        stage1_tablets = batch_data[:6]
        stage1_pass, stage1_info = tester.stage1_test(stage1_tablets)
        print(f"  第一阶段 (前6个药片): {stage1_info}")
        
        # 模拟第二阶段测试
        stage2_tablets = batch_data[:12]
        stage2_pass, stage2_info = tester.stage2_test(stage2_tablets)
        print(f"  第二阶段 (前12个药片): {stage2_info}")
        
        # 模拟第三阶段测试
        stage3_tablets = batch_data[:24]
        stage3_pass, stage3_info = tester.stage3_test(stage3_tablets)
        print(f"  第三阶段 (全部24个药片): {stage3_info}")
        
        print()

if __name__ == "__main__":
    # 测试现有批次
    test_existing_batches()
    
    # 分析失败模式
    analyze_stage_failure_patterns()