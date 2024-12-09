import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.last_x = None

        assert (
            input_dim % num_heads == 0
        ), "Input dimension must be divisible by number of heads"

        # 定义多头的线性变换层
        self.head_dim = input_dim // num_heads
        self.query = nn.Linear(input_dim, num_heads * self.head_dim)
        self.key = nn.Linear(input_dim, num_heads * self.head_dim)
        self.value = nn.Linear(input_dim, num_heads * self.head_dim)
        self.out = nn.Linear(num_heads * self.head_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # 计算 Q, K, V
        Q = self.query(x).view(
            batch_size, self.num_heads, self.head_dim
        )  # Shape: (batch_size, num_heads, head_dim)
        if self.last_x == None:
            K = self.key(x).view(
                batch_size, self.num_heads, self.head_dim
            )  # Shape: (batch_size, num_heads, head_dim)
            V = self.value(x).view(
                batch_size, self.num_heads, self.head_dim
            )  # Shape: (batch_size, num_heads, head_dim)
        else:
            K = self.key(self.last_x).view(
                batch_size, self.num_heads, self.head_dim
            )  # Shape: (batch_size, num_heads, head_dim)
            V = self.value(self.last_x).view(
                batch_size, self.num_heads, self.head_dim
            )  # Shape: (batch_size, num_heads, head_dim)
        # 计算注意力分数（点积）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # Shape: (batch_size, num_heads, num_heads)

        # 计算注意力权重
        attention_weights = F.softmax(
            scores, dim=-1
        )  # Softmax over the last dimension (num_heads)

        # 计算加权的值
        out = torch.matmul(
            attention_weights, V
        )  # Shape: (batch_size, num_heads, head_dim)

        # 合并多头结果并通过线性变换得到输出
        out = out.view(
            batch_size, self.num_heads * self.head_dim
        )  # Shape: (batch_size, num_heads * head_dim)
        out = self.out(out)  # Final output
        self.last_x = x
        return out, attention_weights


class AttentionWithMLP(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(AttentionWithMLP, self).__init__()
        # 使用多头注意力
        self.multihead_attention = MultiHeadAttention(input_dim, num_heads)
        # 增强的MLP结构
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, input_dim)
        )

    def forward(self, x):
        # 获取多头注意力输出
        attention_output, attention_weights = self.multihead_attention(x)
        # 通过MLP进一步映射输出
        output = self.mlp(attention_output)
        return output, attention_weights

if __name__ == "__main__":
    # 创建一个输入大小为32维的随机数据
    input_data = torch.randn(1, 32)  # 1个样本，32个特征

    # 实例化增强注意力模型
    attention_model = AttentionWithMLP(input_dim=32)

    # 计算注意力权重和输出
    output, normalized_weights = attention_model(input_data)

    # 输出维度
    print("Output shape:", output.shape,output)  # Expected (1, 32)
    print(
        "Normalized Weights shape:", normalized_weights.shape, normalized_weights
    )  # Expected (1, 4, 4)
