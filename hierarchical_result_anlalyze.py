import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
def plot_bars(data,scenes,agents,metrices,Labels):
    x_labels = []
    figsize = (18,4)
    
    total_width= 0.4

    # 对比模型数量
    num_methods = len(agents)

    # 指标数量
    num_metrics = len(metrices)

    # 每种类型的柱状图宽度
    width = total_width / num_methods

    # 对比实验场景数量
    scenes_num = len(scenes)
    for scene in scenes:
        x_labels.extend(["",scene])
    x = np.arange(scenes_num)
    fig, axs = plt.subplots(1, num_metrics, figsize=figsize)
    color_name = 'Set3'
    # color_idx = random.sample(range(1, 10), num_methods)
    color_idx = [6,4,5,3]
    colors = plt.get_cmap(color_name)(color_idx) # 从色组里选择颜色，我选择的是select2
    metrics_bottom = {'success_flag':60,
                      "avg_dist":2.3,
                      "collision":0,
                      "epi_len":50
                      }
    titles = {'success_flag':"Success Rate(%)",
              "avg_dist":"Average Safe Distance Throughout An Episode",
              "collision":"Collision Rate(%)",
              "epi_len":"Average Path Length"
              }
    if num_methods %2 == 0:
        offset_distance = np.arange(-width*num_methods/2-width,width*num_methods/2,width)
    else:
        offset_distance = np.arange(-(num_methods-1)/2*width-width*3/2,(num_methods-1)/2*width,width)

    # 绘制子图
    for i in range(num_metrics):
        for j in range(num_methods):
            data_array = []
            for scene in scenes:
                data_array.append(data[scene][agents[j]][metrices[i]]['mean'])
            if metrices[i] == "success_flag":
                data_array = np.array(data_array)*100
            else:
                data_array = np.array(data_array)
            axs[i].bar(x+offset_distance[j],data_array-metrics_bottom[metrices[i]], width=width,label=Labels[j], color=colors[j], bottom=metrics_bottom[metrices[i]])
        axs[i].set_xticklabels(x_labels)
        axs[i].set_title(titles[metrics[i]])

    # 设置图例（图例只在第一个子图中显示，其他子图隐藏图例）
    for ax in axs.flat:
        ax.legend().remove()

    # 获取所有线条和标签
    lines, labels = fig.axes[-1].get_legend_handles_labels()

    # 创建一个全局图例，放在最底部并水平铺开
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=5)

    # 显示图像
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 调整布局，留出底部空间
    plt.savefig("Hierarchical_Performence_Compare.jpg",dpi=300)

def read_and_compute(file_path,Agent_names,relax_ending_condition = False):
    # 返回各列的均值及方差
    try:
        # 读取数据文件（假设是以空格、逗号或制表符分隔的格式）
        data = pd.read_csv(file_path)
        if relax_ending_condition:
            for agent in Agent_names:
                data.loc[data[f"epi_len_{agent}"]==2000,f"success_flag_{agent}"] = 1
        # 计算每列的均值和方差
        results = {}
        for column in data.columns:
            mean_val = np.mean(data[column])
            var_val = np.var(data[column])
            results[column] = {'mean': mean_val, 'variance': var_val}
        
        # 打印结果
        # for col, stats in results.items():
        #     print(f"Column: {col}, Mean: {stats['mean']:.4f}, Variance: {stats['variance']:.4f}")
        
        return results
    
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return None

# 示例用法
task_type = "hierarchical"
Agent_names = ["Nav","Hierarchical",'KURL','Rule_Based']
Labels = ["Nav","Hierarchical",'KURL','Rule_Based']
scenes_name = ["Dense"]
# 3_28_22_14_29 3_28_22_15_27 3_28_22_15_28 3_28_22_15_22 3_28_22_14_29 
# 4_2_8_58_11 4_2_8_59_35 4_2_9_0_6 4_2_9_0_32 4_2_9_12_4
records_path = ["4_2_9_0_32"]
metrics = ['success_flag',"avg_dist","collision_steps","epi_len","lost_steps","track_steps","max_lost_step","refind_count","max_track_step"]
records_results = {}
for i in range(len(records_path)):
    records_results[scenes_name[i]] = {}
    file_path = f"test_records/{task_type}/{records_path[i]}/result_{task_type}_{records_path[i]}.csv" 
    # 取出文档中每个agent每个指标的均值及方差
    data = read_and_compute(file_path,Agent_names,relax_ending_condition=True)
    results = {}
    for agent in Agent_names:
        results[agent] = {}
        for metrice in metrics:
            metrice_key = metrice+'_'+agent
            results[agent][metrice] = data[metrice_key]
    records_results[scenes_name[i]] = results
for agent in Agent_names:
    print("*"*10+agent+"*"*10)
    print(records_results['Dense'][agent])
    

# plot_bars(records_results,scenes_name,Agent_names,metrics,Labels)
    


