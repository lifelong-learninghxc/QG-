import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载数据 (使用 seaborn 内置的泰坦尼克数据集)
titanic = sns.load_dataset('titanic')

# 2. 特征工程与预处理
# 选择核心特征：客舱等级、性别、年龄、兄弟姐妹/配偶数、父母/子女数、票价
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
X = titanic[features].copy()
y = titanic['survived']

# 处理缺失值：使用年龄的中位数填充缺失的年龄
X['age'] = X['age'].fillna(X['age'].median())

# 分类变量编码：将性别独热编码 (Male变成0/1)
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 构建并训练决策树模型
# 设置 max_depth=5 防止树生长得过深导致过拟合
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 5. 预测与评估
y_pred = clf.predict(X_test)

print(f"模型准确率: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("分类报告:")
print(classification_report(y_test, y_pred))