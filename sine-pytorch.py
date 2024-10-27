import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# GPU
if torch.cuda.is_available():
    device = torch.device("cuda")


# 神经网络定义
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out

# 参数定义
input_size = 1
output_size = 1
num_epochs = 1000
learning_rate = 0.01

# 生成训练数据
x_train = np.linspace(-2*np.pi, 2*np.pi, 100)
y_train = np.sin(x_train)

x_train = torch.Tensor(x_train).view(-1, 1).to(device)    # 转为torch的张量格式，列向量
y_train = torch.Tensor(y_train).view(-1, 1).to(device)

# 带入神经网络参数
model = Net(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练神经网络
for epoch in range(num_epochs):
    # 前向传播
    output = model(x_train)
    loss = criterion(output, y_train)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 ++ 0:
        print(f"Epoch: [ {epoch+1} / {num_epochs}] , loss: {loss.item(): .4f}")

# 生成测试数据
x_test = np.linspace(-2*np.pi, 2*np.pi, 1000)
x_test = torch.Tensor(x_test).view(-1,1).to(device)

# 预测结果
model.eval()
with torch.no_grad():
    y_pred = model(x_test)

# 可视化
plt.figure(figsize=(10, 6))
x_plot = np.linspace(-2*np.pi, 2*np.pi, 1000)
y_plot = np.sin(x_plot)
plt.plot(x_plot, y_plot, color='g', label='sin(x)')  # 绘制sin函数曲线
plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), color='b', label='Training data')  # 绘制训练数据点
plt.scatter(x_test.cpu().numpy(), y_pred.cpu().numpy(), color='r', label='Predicted points')  # 绘制预测点
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting sin(x) with Neural Network')
plt.legend()
plt.show()