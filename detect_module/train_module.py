import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import datetime

# 设置路径
data_dir = "./detect_module/labeled_images/real"
img_height, img_width = 250, 250  # 图像大小
time = datetime.datetime.now()  # 获取当前时间
log_dir = "./detect_module/detection_record"
log_path = os.path.join(log_dir,"detection_{}_{}_{}_{}".format(
    time.month, time.day, time.hour, time.minute, time.second))
if os.path.exists(log_path) == False:
    os.mkdir(log_path)
# 图像预处理和数据增强
transform = transforms.Compose(
    [
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for filename in os.listdir(data_dir):
            img = Image.open(os.path.join(data_dir, filename)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            self.images.append(img)
            label = 1 if "target" in filename else 0
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


# 读取数据集
dataset = CustomDataset(data_dir, transform)

# 使用数据生成器
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 构建模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (img_height // 2) * (img_width // 2), 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * (img_height // 2) * (img_width // 2))
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 记录损失
losses = []
kf = KFold(n_splits=5)  # k轮交叉验证
epochs = 100
# 训练模型
model.train()
for epoch in range(epochs):
    epoch_loss = 0  # 每个epoch的损失初始化
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # 计算并记录平均损失
    average_loss = epoch_loss / len(data_loader)
    losses.append(average_loss)
    print("epoch:", epoch, "loss:", average_loss)

    # 每2个epoch进行k轮交叉验证
    if (epoch + 1) % 2 == 0:
        y_true = []
        y_pred_prob = []

        for train_idx, val_idx in kf.split(np.arange(len(dataset))):
            # 划分训练和验证集
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader_cv = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True
            )
            val_loader_cv = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # 在每个交叉验证折上训练模型
            model.train()
            for inputs, labels in train_loader_cv:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

            # 验证模型
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader_cv:
                    inputs = inputs.to(device)
                    outputs = model(inputs)

                    # 确保输出为一维
                    outputs = outputs.squeeze()
                    if outputs.ndim == 0:  # 如果是0维张量，转换为一维
                        outputs = outputs.unsqueeze(0)

                    y_pred_prob.extend(outputs.cpu().numpy().tolist())
                    y_true.extend(labels.numpy())

        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        print(f"Cross-validation ROC AUC after epoch {epoch + 1}: {roc_auc:.2f}")
        # 绘制并保存ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")

        # 保存ROC曲线图像
        plt.savefig(f"{log_path}/roc_curve_epoch_{epoch + 1}.jpg")  # 保存文件名
        plt.close()  # 关闭当前图形以释放内存
# 绘制损失曲线
plt.figure()
plt.plot(range(1, 1+epochs), losses, marker="o", linestyle="-", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig(f"{log_path}/loss_curve.jpg")  # 保存损失曲线图像
# plt.show()

# 保存模型
torch.save(model.state_dict(), f"{log_path}/object_detection_model.pth")
