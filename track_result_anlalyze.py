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
    metrics_bottom = {'reward':-0.5,
                      "epi_len":1500,
                      "angle_acc":0.020,
                      "distance_acc":0.5
                      }
    titles = {'reward':"Average Reward Per Step",
              "epi_len":"Average Tracking Steps Per Episode",
              "angle_acc":"Average Relative Angle Per Episode",
              "distance_acc":"Average Relative Distance Per Epissode"
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
            if metrices[i] == "angle_acc":
                data_array = np.array(data_array)*2*np.pi/360
            else:
                data_array = np.array(data_array)
            if metrices[i] == "epi_len":
                data_array *= 2
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
    plt.savefig("跟踪性能对比.jpg",dpi=300)
    print("图片已保存")

def read_and_compute(file_path,relax_ending_condition = False):
    # 返回各列的均值及方差
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
        # for col, stats in results.items():
        #     print(f"Column: {col}, Mean: {stats['mean']:.4f}, Variance: {stats['variance']:.4f}")
        
        return results
    
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return None

# 示例用法
task_type = "track"
Agent_names = ["RNN_MHA",'RNN','MHA','MLP_ONLY']
Labels = ["RNN_MHA",'RNN','MHA','MLP_ONLY']
scenes_name = ["Dense","Medium","Sparse"]
# records_path = ["3_26_11_27_3","3_26_11_29_24","3_26_11_29_59"]
records_path = ["4_21_8_56_43","4_21_8_55_57","4_21_8_54_4"]
metrics = ['reward',"epi_len","angle_acc","distance_acc"]
records_results = {}
for i in range(len(records_path)):
    records_results[scenes_name[i]] = {}
    file_path = f"/media/hp/新加卷/XNW/Hiearchical_RL/test_records/{task_type}/{records_path[i]}/result_{task_type}_{records_path[i]}.csv" 
    # 取出文档中每个agent每个指标的均值及方差
    data = read_and_compute(file_path)
    results = {}
    for agent in Agent_names:
        results[agent] = {}
        for metrice in metrics:
            metrice_key = metrice+'_'+agent
            results[agent][metrice] = data[metrice_key]
    records_results[scenes_name[i]] = results
for scene in records_results.keys():
    print(f"**********{scene}**********")
    for agent in records_results[scene].keys():
        print(f"{agent}:")
        for metrice in records_results[scene][agent].keys():
            print(f"________{metrice}________")
            print(f"average:{records_results[scene][agent][metrice]['mean']},variance:{records_results[scene][agent][metrice]['variance']}")

# print(records_results)
plot_bars(records_results,scenes_name,Agent_names,metrics,Labels)
    


