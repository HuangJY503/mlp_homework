import numpy as np
import matplotlib.pyplot as plt
# 生成训练数据
np.random.seed(0)
x_train = np.random.uniform(-4, 4, 160)
y_train = 2 * (1 - x_train + 2 * x_train**2) * np.exp(-x_train**2 / 2) + np.random.normal(0, 0.1, 160)
#print(x_train)

# 定义MLP网络
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, x):
        self.z = np.dot(x, self.W1) + self.b1
        self.a = np.tanh(self.z)
        self.z2 = np.dot(self.a, self.W2) + self.b2
        self.y_hat = self.z2

    def backward(self, x, y):
        m = x.shape[0]
        error = self.y_hat - y
        dW2 = (1 / m) * np.dot(self.a.T, error)
        db2 = (1 / m) * np.sum(error, axis=0, keepdims=True)
        error2 = np.dot(error, self.W2.T) * (1 - np.power(self.a, 2))
        dW1 = (1 / m) * np.dot(x.T, error2)
        db1 = (1 / m) * np.sum(error2, axis=0, keepdims=True)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

# 训练网络
def train_mlp(hidden_size, learning_rate, num_train_samples,epoches = 1000):
    input_size = 1
    output_size = 1
    mlp = MLP(input_size, hidden_size, output_size, learning_rate)
    x_train_sampled = x_train[:num_train_samples]
    y_train_sampled = y_train[:num_train_samples]
    mse_list = []
    for i in range(epoches):
        mlp.forward(x_train_sampled.reshape(-1, 1))
        mlp.backward(x_train_sampled.reshape(-1, 1), y_train_sampled.reshape(-1, 1))
        #计算每次的训练误差，保存到一个list
        mse = np.mean((mlp.y_hat - y_train_sampled.reshape(-1, 1))**2)
        mse_list.append(mse)
    return mlp

# 运行得到训练结果
def run_mlp(mlp,ax, num_train_samples,if_visualize = 1):
    x = np.linspace(-4, 4, 160)
    y = 2 * (1 - x + 2 * x**2) * np.exp(-x**2 / 2)
    y_pred = np.zeros_like(x)
    for i in range(len(x)):
        mlp.forward(x[i].reshape(-1, 1))
        y_pred[i] = mlp.y_hat
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)

    if(if_visualize):
        ax.plot(x, y_pred, label='MLP Prediction')
        ax.legend()
        #plt.show()
    return mse
#draw picture
fig, ax = plt.subplots()  # 创建图形和轴对象
fig2, ax2 = plt.subplots()  # 创建图形和轴对象


# 测试不同训练样本数、隐含单元个数、学习率的影响
num_train_samples_list = [15, 30, 45, 60, 80]
hidden_size_list = [2, 4, 8, 16]
learning_rate_list = [0.01, 0.05, 0.1, 0.2]


# 绘制'True Function'和'Training Data'
x = np.linspace(-4, 4, 160)
y = 2 * (1 - x + 2 * x ** 2) * np.exp(-x ** 2 / 2)
ax.plot(x, y, label='True Function')
ax.scatter(x_train, y_train, label='Training Data')
#
# # #inpact of num_train_samples
select =3 #1: num_train_samples 2:hidden_size 3:learning_rate 0:single test
if(select == 1):
    for num_train_samples in num_train_samples_list:
        hidden_size = 16
        learning_rate = 0.1
        mlp = train_mlp(hidden_size, learning_rate, num_train_samples)
        x = np.linspace(-4, 4, 160)
        y_pred = np.zeros_like(x)
        for i in range(len(x)):
            mlp.forward(x[i].reshape(-1, 1))
            y_pred[i] = mlp.y_hat
        # 计算均方误差
        mse = np.mean((y_pred - y) ** 2)
        print(mse)
        ax.plot(x, y_pred, label='MLP Prediction (num_train_samples=%d)' % num_train_samples)
#
# #inpact of hidden_size
elif(select == 2):
    for hidden_size in hidden_size_list:
        num_train_samples = 80
        learning_rate = 0.1
        mlp = train_mlp(hidden_size, learning_rate, num_train_samples)
        x = np.linspace(-4, 4, 160)
        y_pred = np.zeros_like(x)
        for i in range(len(x)):
            mlp.forward(x[i].reshape(-1, 1))
            y_pred[i] = mlp.y_hat
        # 计算均方误差
        mse = np.mean((y_pred - y) ** 2)
        print(mse)
        ax.plot(x, y_pred, label='MLP Prediction (hidden_size=%d)' % hidden_size)

#inpact of learning_rate
elif(select == 3):
    for learning_rate in learning_rate_list:
        num_train_samples = 80
        hidden_size = 16
        mlp = train_mlp(hidden_size, learning_rate, num_train_samples)
        x = np.linspace(-4, 4, 160)
        y_pred = np.zeros_like(x)
        for i in range(len(x)):
            mlp.forward(x[i].reshape(-1, 1))
            y_pred[i] = mlp.y_hat
        # 计算均方误差
        mse = np.mean((y_pred - y) ** 2)
        print(mse)
        ax.plot(x, y_pred, label='MLP Prediction (learning_rate=%f)' % learning_rate)
elif(select ==0):
# single test
    hidden_size = 16
    learning_rate = 0.1
    num_train_samples = 80
    mlp,mse_list = train_mlp(hidden_size, learning_rate, num_train_samples)
    run_mlp(mlp, ax,num_train_samples)
    ax2.plot(mse_list,label = 'MSE')

ax.legend()  # 显示图例
#ax1坐标轴设置
ax.set_xlabel('X')
ax.set_ylabel('Y')

ax2.legend()  # 显示图例
#ax2坐标轴设置
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')

plt.show()  # 显示图形


