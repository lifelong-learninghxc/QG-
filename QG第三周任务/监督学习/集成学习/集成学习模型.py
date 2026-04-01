import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# 导入我们今天的三位主角：单一树、随机森林(Bagging)、梯度提升(Boosting)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ================= 1. 数据加载与预处理 (复用你之前的完美逻辑) =================
titanic = sns.load_dataset('titanic')
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
X = titanic[features].copy()
y = titanic['survived']

# 填充缺失值与独热编码
X['age'] = X['age'].fillna(X['age'].median())
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= 2. 构建模型字典，方便批量训练与评估 =================
models = {
    "1. 单一决策树 (Baseline)": DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42),
    "2. 随机森林 (Bagging)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "3. 梯度提升树 (Boosting)": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# ================= 3. 训练并打印性能分析 =================
for name, model in models.items():
    print(f"{'='*10} 正在评估: {name} {'='*10}")
    # 训练模型
    model.fit(X_train, y_train)
    # 预测结果
    y_pred = model.predict(X_test)
    
    # 计算并打印指标
    acc = accuracy_score(y_test, y_pred)
    print(f"准确率 (Accuracy): {acc:.4f}\n")
    print("详细分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n")