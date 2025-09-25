"""
药品溶解度数据预处理模块
用于处理从Word文档中提取的溶解度测试数据
"""

import pandas as pd
import numpy as np
from docx import Document

def extract_dissolution_data(docx_path):
    """
    从Word文档中提取溶解度数据
    
    Args:
        docx_path (str): Word文档路径
        
    Returns:
        pd.DataFrame: 包含批次号、药片编号和溶解度的数据框
    """
    doc = Document(docx_path)
    
    # 提取表格数据
    dissolution_data = []
    
    for table in doc.tables:
        # 跳过表头
        for i, row in enumerate(table.rows):
            if i == 0:  # 跳过表头
                continue
                
            cells = [cell.text.strip() for cell in row.cells]
            if len(cells) >= 3 and cells[0] and cells[1] and cells[2]:
                try:
                    batch_id = cells[0]
                    tablet_id = int(cells[1])
                    dissolution = float(cells[2])
                    dissolution_data.append({
                        'batch_id': batch_id,
                        'tablet_id': tablet_id,
                        'dissolution': dissolution
                    })
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(dissolution_data)

def load_sample_data():
    """
    加载示例数据（从题目中提取的数据）
    
    Returns:
        pd.DataFrame: 溶解度数据
    """
    # 基于题目中的数据创建数据集
    data = []
    
    # B001批次数据
    b001_values = [85, 83, 87, 81, 86, 84, 85, 84, 85, 82, 83, 84, 
                   81, 85, 87, 78, 86, 84, 88, 84, 80, 83, 85, 82]
    
    # B002批次数据
    b002_values = [80, 82, 80, 81, 83, 78, 81, 81, 81, 79, 82, 80,
                   79, 81, 80, 83, 76, 75, 80, 79, 81, 78, 82, 78]
    
    # B003批次数据
    b003_values = [80, 85, 79, 82, 77, 75, 88, 83, 84, 78, 86, 83,
                   87, 86, 84, 81, 82, 83, 86, 84, 85, 81, 81, 83]
    
    # 组织数据
    for i, value in enumerate(b001_values):
        data.append({'batch_id': 'B001', 'tablet_id': i+1, 'dissolution': value})
    
    for i, value in enumerate(b002_values):
        data.append({'batch_id': 'B002', 'tablet_id': i+1, 'dissolution': value})
        
    for i, value in enumerate(b003_values):
        data.append({'batch_id': 'B003', 'tablet_id': i+1, 'dissolution': value})
    
    return pd.DataFrame(data)

def get_batch_statistics(df):
    """
    计算各批次的统计信息
    
    Args:
        df (pd.DataFrame): 溶解度数据
        
    Returns:
        pd.DataFrame: 各批次统计信息
    """
    stats = df.groupby('batch_id')['dissolution'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    
    return stats

def save_data_to_csv(df, filename):
    """
    将数据保存为CSV文件
    
    Args:
        df (pd.DataFrame): 要保存的数据
        filename (str): 文件名
    """
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"数据已保存到 {filename}")

if __name__ == "__main__":
    # 加载数据
    print("正在加载溶解度数据...")
    df = load_sample_data()
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"包含 {df['batch_id'].nunique()} 个批次")
    
    # 显示数据概览
    print("\n数据概览:")
    print(df.head(10))
    
    # 计算统计信息
    print("\n各批次统计信息:")
    stats = get_batch_statistics(df)
    print(stats)
    
    # 保存数据
    save_data_to_csv(df, '溶解度数据.csv')
    save_data_to_csv(stats, '批次统计.csv')
    
    print("\n数据预处理完成！")