import numpy as np
from sklearn import datasets

# ==========================================
# 1. 数据准备与预处理
# ==========================================

# 加载原始数据
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 从零实现：训练集/测试集划分
def train_test_split_scratch(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# 从零实现：Z-score 标准化
def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1e-8  # 防止除以零
    return (X_train - mean) / std, (X_test - mean) / std

# 从零实现：One-Hot 编码
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

X_train, X_test, y_train, y_test = train_test_split_scratch(X, y)
X_train_std, X_test_std = standardize(X_train, X_test)
y_train_oh = one_hot_encode(y_train, num_classes=3)

# ==========================================
# 2. 构建 Softmax 回归模型
# ==========================================

class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs

    def softmax(self, z):
        # 减去最大值防止指数爆炸 (Numerical Stability)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        m, n = X.shape
        num_classes = y.shape[1]
        
        # 初始化参数
        self.W = np.zeros((n, num_classes))
        self.b = np.zeros((1, num_classes))

        # 梯度下降优化
        for epoch in range(self.epochs):
            z = np.dot(X, self.W) + self.b
            y_hat = self.softmax(z)

            # 计算梯度
            dw = (1 / m) * np.dot(X.T, (y_hat - y))
            db = (1 / m) * np.sum(y_hat - y, axis=0, keepdims=True)

            # 更新权重
            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_hat = self.softmax(z)
        return np.argmax(y_hat, axis=1)

# ==========================================
# 3. 训练与模型评估
# ==========================================

model = SoftmaxRegression(learning_rate=0.1, epochs=2000)
model.fit(X_train_std, y_train_oh)
predictions = model.predict(X_test_std)

# 计算准确率和混淆矩阵
def evaluate(y_true, y_pred, num_classes=3):
    accuracy = np.mean(y_true == y_pred)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1
    return accuracy, confusion_matrix

acc, conf_mat = evaluate(y_test, predictions)
print(f"测试集准确率: {acc * 100:.2f}%")
print("混淆矩阵:\n", conf_mat)