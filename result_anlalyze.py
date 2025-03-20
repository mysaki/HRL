import numpy as np
import pandas as pd

def read_and_compute(file_path,relax_ending_condition):
    try:
        # 读取数据文件（假设是以空格、逗号或制表符分隔的格式）
        data = pd.read_csv(file_path)
        if relax_ending_condition:
            data.loc[data["epi_len_Hierarchical"]==2000,"success_flag_Hierarchical"] = 1
            data.loc[data["epi_len_Rule_Based"]==2000,"success_flag_Rule_Based"] = 1
        # 计算每列的均值和方差
        results = {}
        for column in data.columns:
            mean_val = np.mean(data[column])
            var_val = np.var(data[column])
            results[column] = {'mean': mean_val, 'variance': var_val}
        
        # 打印结果
        for col, stats in results.items():
            print(f"Column: {col}, Mean: {stats['mean']:.4f}, Variance: {stats['variance']:.4f}")
        
        return results
    
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return None

# 示例用法
date="3_16_13_19_14"
file_path = f"test_records/hierarchical/{date}/result_hierarchical_{date}.csv"  # 这里可以替换成你的文件路径
relax_ending_condition = True
read_and_compute(file_path,relax_ending_condition)