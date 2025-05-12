import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple
import csv
LEARNING_RATE = 0.001

class Es_Net(nn.Module):
    def __init__(self,
     input_size,
     output_size,
     experience_replay,
     loss_fn,
     ):
        super(Es_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.experience_replay=experience_replay
        # 定义损失函数
        self.loss_fn = loss_fn
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # 训练
    def train(self,net,batch_size):
        if len(self.experience_replay) < batch_size:
            return
        experiences = self.experience_replay.sample(batch_size)
        last_pos = torch.tensor(batch.last_pos, dtype=torch.float32)
        next_pos = torch.tensor(batch.next_pos, dtype=torch.float32)


        predict_pos = self.forward(last_pos)
        self.optimizer= optim.Adam(net.parameters(), lr=LEARNING_RATE)
        loss = self.loss_fn(predict_pos,next_pos)
        with open('./train_es.csv','a') as f:
            writer=csv.writer(f)
            writer.writerow([loss.detach().cpu().numpy()])
        print("loss:",loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__=='__main__':
    # 设置超参数
    BATCH_SIZE = 64
    CAPACITY = 10000
    LEARNING_RATE = 0.001
    GAMMA = 0.99

    # 初始化神经网络和经验池
    input_size = 4  # 输入状态的维度
    output_size = 2  # 输出动作的数量
    es_net = Es_Net(input_size, output_size)
    optimizer = optim.Adam(es_net.parameters(), lr=LEARNING_RATE)
    experience_replay = ExperienceReplay(CAPACITY)

    # 定义损失函数
    loss_fn = nn.MSELoss()
    # 示例：收集观察数据，存储到经验池中，并训练神经网络
    for i in range(1000):
        # 模拟环境中的动作选择和状态转换
        state = [random.random(), random.random(), random.random(), random.random()]  # 示例状态
        action = random.randint(0, 1)  # 示例动作
        next_state = [random.random(), random.random(), random.random(), random.random()]  # 示例下一个状态
        reward = random.random()  # 示例奖励值

        # 存储观察数据到经验池


        # 使用经验池中的数据训练神经网络
        es_net.train(BATCH_SIZE)
