import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 设置路径
model_path = "detection_record/detection_11_19_20_4/object_detection_model.pth"  # 加载模型的路径
data_dir = "labeled_images/mix"  # 验证图片所在文件夹路径
log_path = "detection_record/detection_11_19_20_4"
test_type = "mix"
img_height, img_width = 250, 250  # 图像大小

# 图像预处理
transform = transforms.Compose(
    [
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ]
)


# 加载模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (img_height // 2) * (img_width // 2), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * (img_height // 2) * (img_width // 2))
        x = torch.sigmoid(self.fc2(nn.functional.relu(self.fc1(x))))
        return x


# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))  # 确保加载到正确设备
model.eval()

# 验证图片并输出结果
results = []
y_true = []  # 真实标签
y_pred = []  # 预测标签

# 检查数据目录是否存在
if not os.path.exists(data_dir):
    print(f"Directory {data_dir} does not exist.")
else:
    for filename in os.listdir(data_dir):
        img_path = os.path.join(data_dir, filename)

        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)  # 增加batch维度

            with torch.no_grad():
                output = model(img)
                probability = output.item()  # 转换为标量
                pred_label = 1 if probability > 0.5 else 0  # 二分类判断
                true_label = 1 if "target" in filename else 0
            results.append((filename, probability, pred_label,true_label))
            y_true.append(true_label)  # 添加真实标签
            y_pred.append(pred_label)  # 添加预测标签
            print(
                f"Image: {filename}, Probability: {probability:.4f}, Predicted Label: {pred_label}, True Label: {true_label}"
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# 可选：保存结果到文件
with open(f"{log_path}/{test_type}_validation_results.txt", "w") as f:
    for filename, probability, pred_label, true_label in results:
        f.write(
            f"{filename}, Probability: {probability:.4f}, Predicted Label: {pred_label}, True Label: {true_label}\n"
        )

# 计算指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# 输出指标
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 可选：保存指标到文件
with open(f"{log_path}/{test_type}_validation_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
