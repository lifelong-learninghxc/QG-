import numpy as np
import pandas as pd
from collections import Counter

# ==========================================
# 1. 数据加载与预处理 (直接从URL获取)
# ==========================================
def load_iris_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    print("正在从网络下载 Iris 数据集...")
    # Iris 数据集没有表头，我们手动指定
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(url, header=None, names=columns)
    
    # 提取特征 (X) 和 真实标签 (y)
    X = df.iloc[:, :-1].values
    y_str = df.iloc[:, -1].values
    
    # 将真实的字符串标签转换为数字 (仅用于最后的评估对比，不参与聚类训练)
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y_true = np.array([species_map[s] for s in y_str])
    
    print("数据加载完毕！\n")
    return X, y_true

# ==========================================
# 2. K-Means 聚类算法核心实现
# ==========================================
def kmeans(X, k=3, max_iters=100, random_state=42):
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    # Step 1: 随机初始化 K 个簇心 (从现有数据点中随机挑选)
    initial_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[initial_indices]
    
    labels = np.zeros(n_samples)
    
    for iteration in range(max_iters):
        # Step 2: 计算每个样本到各个簇心的欧氏距离
        # 巧妙利用 numpy 广播机制: X 维度 (150, 1, 4), centroids 维度 (3, 4) -> 距离矩阵 (150, 3)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # Step 3: 将每个样本分配给距离最近的簇心
        new_labels = np.argmin(distances, axis=1)
        
        # Step 4: 更新簇心 (计算分配到该簇的所有样本的均值)
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            # 获取属于第 i 个簇的所有样本
            cluster_points = X[new_labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # 如果某个簇空了，保留原簇心 (或者可以重新随机初始化)
                new_centroids[i] = centroids[i]
                
        # 检查是否收敛 (簇心不再变化)
        if np.all(centroids == new_centroids):
            print(f"K-Means 算法在第 {iteration + 1} 次迭代后收敛。")
            break
            
        centroids = new_centroids
        labels = new_labels
        
    return labels, centroids

# ==========================================
# 3. 模型评估函数
# ==========================================
def evaluate_kmeans(X, labels_pred, centroids, y_true):
    print("\n--- K-Means 聚类模型评估 ---")
    
    # 评估指标 1：簇内误差平方和 WCSS (Inertia) - 无监督评估
    wcss = 0
    k = len(centroids)
    for i in range(k):
        cluster_points = X[labels_pred == i]
        wcss += np.sum((cluster_points - centroids[i])**2)
    print(f"WCSS (簇内误差平方和 / Inertia): {wcss:.4f}")
    
    # 评估指标 2：聚类准确率 (需要与真实标签对齐) - 仅限已知真实标签的数据集
    # 因为 K-Means 给出的标签 (0, 1, 2) 是随机分配的，可能和 y_true (0, 1, 2) 的含义不一致
    # 我们需要通过多数投票法，将预测的簇编号映射到最可能的真实类别
    mapped_labels = np.zeros_like(labels_pred)
    for i in range(k):
        # 找到被分到簇 i 的所有样本的真实标签
        true_labels_in_cluster = y_true[labels_pred == i]
        if len(true_labels_in_cluster) > 0:
            # 找到数量最多的那个真实标签作为该簇的代表
            most_common = Counter(true_labels_in_cluster).most_common(1)[0][0]
            mapped_labels[labels_pred == i] = most_common
            
    accuracy = np.mean(mapped_labels == y_true)
    print(f"聚类纯度/对齐准确率 (Clustering Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

# 执行主流程
if __name__ == "__main__":
    X, y_true = load_iris_data()
    
    # 鸢尾花已知有3种，所以设定 k=3
    labels_pred, centroids = kmeans(X, k=3)
    
    evaluate_kmeans(X, labels_pred, centroids, y_true)