import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 数据加载与预处理
# 假设我们下载了著名的 UCI SMS Spam Collection 数据集
# 数据通常包含两列：标签 (ham/spam) 和 文本内容 (message)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 将文本标签转换为二分类数值 (ham: 0, spam: 1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label_num']

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 3. 文本向量化 (词袋模型 Bag of Words)
# CountVectorizer 会将文本切分成单词，并统计每个单词在每条短信中出现的次数
vectorizer = CountVectorizer(stop_words='english') # 自动去除英文停用词(如 is, the, and)
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

# 4. 构建并训练多项式朴素贝叶斯模型
# alpha=1.0 表示使用拉普拉斯平滑 (Laplace Smoothing)
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_dtm, y_train)

# 5. 模型预测与评估
y_pred = nb_model.predict(X_test_dtm)

print(f"测试集准确率: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=['正常短信(Ham)', '垃圾短信(Spam)']))

print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))