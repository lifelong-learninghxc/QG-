import numpy as np
import pandas as pd

# ==========================================
# 1. 数据加载与预处理 (直接从URL获取)
# ==========================================
def load_and_preprocess_data():
    # 直接使用 pandas 从官方 URL 读取 CSV 数据
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    print("正在从网络下载红酒数据集，请稍候...")
    df = pd.read_csv(url, sep=';')
    
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 特征标准化 (Z-score 归一化)
    # 必须只使用训练集的参数对测试集归一化，防止数据穿越
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # 增加偏置项截距 (x_0 = 1)
    X_train_norm = np.c_[np.ones(X_train_norm.shape[0]), X_train_norm]
    X_test_norm = np.c_[np.ones(X_test_norm.shape[0]), X_test_norm]
    
    print("数据加载并预处理完毕！\n")
    return X_train_norm, X_test_norm, y_train, y_test

# ==========================================
# 2. 线性回归模型 (预测具体评分)
# ==========================================
def linear_regression(X_train, y_train, X_test):
    # 使用正规方程直接求解解析解: theta = (X^T * X)^(-1) * X^T * y
    theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    y_pred = X_test.dot(theta)
    return y_pred

# ==========================================
# 3. 逻辑回归模型 (区分好坏酒)
# ==========================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X_train, y_train, X_test, lr=0.1, epochs=5000):
    # 根据题目要求，将目标值二值化（quality > 6 为好酒 1，否则为坏酒 0）
    y_train_bin = (y_train > 6).astype(int)
    
    m, n = X_train.shape
    theta = np.zeros(n)
    
    # 使用梯度下降法更新参数
    for _ in range(epochs):
        z = X_train.dot(theta)
        h = sigmoid(z)
        # 计算梯度: X^T * (h - y) / m
        gradient = X_train.T.dot(h - y_train_bin) / m
        theta -= lr * gradient
        
    # 预测概率并转化为类别标签
    y_pred_prob = sigmoid(X_test.dot(theta))
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    return y_pred_bin

# ==========================================
# 4. 模型评估函数
# ==========================================
def evaluate_models(y_true, y_pred_lin, y_pred_log):
    print("--- 线性回归评估 (回归任务) ---")
    mse = np.mean((y_true - y_pred_lin)**2)
    rmse = np.sqrt(mse)
    # R平方 (R-squared)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred_lin)**2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"MSE (均方误差): {mse:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}\n")

    print("--- 逻辑回归评估 (分类任务) ---")
    y_true_bin = (y_true > 6).astype(int)
    accuracy = np.mean(y_true_bin == y_pred_log)
    
    # 混淆矩阵要素
    tp = np.sum((y_pred_log == 1) & (y_true_bin == 1))
    fp = np.sum((y_pred_log == 1) & (y_true_bin == 0))
    fn = np.sum((y_pred_log == 0) & (y_true_bin == 1))
    tn = np.sum((y_pred_log == 0) & (y_true_bin == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy (准确率): {accuracy:.4f}")
    print(f"Precision (精确率): {precision:.4f}")
    print(f"Recall (召回率): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# 执行主流程
if __name__ == "__main__":
    # 现在这里不需要传入本地路径了，函数内部处理了网络请求
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    y_pred_lin = linear_regression(X_train, y_train, X_test)
    y_pred_log = logistic_regression(X_train, y_train, X_test)
    evaluate_models(y_test, y_pred_lin, y_pred_log)