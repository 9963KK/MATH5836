import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 加载数据集
file_path = 'processed.cleveland.data'  # Heart Disease 数据集路径

# 添加列名
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(file_path, names=column_names, header=None, na_values='?')

# 数据预处理 - 检查缺失值并填充
print(df.isnull().sum())  # 检查缺失值
df = df.dropna()  # 删除包含缺失值的行

# 数据分析与可视化
# 查看数据基本信息
df.info()

# 绘制目标变量的分布
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Presence (0 = No, 1 = Yes)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# 绘制特征之间的相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

# 模型构建 - 将 'target' 作为目标变量，其他特征作为输入
X = df.drop(columns=['target'])
y = df['target']

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练决策树分类器
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 评估决策树分类器
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
print(f"决策树 - 准确率: {accuracy_dt:.2f}, F1 分数: {f1_dt:.2f}")

# 训练随机森林分类器
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 评估随机森林分类器
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
print(f"随机森林 - 准确率: {accuracy_rf:.2f}, F1 分数: {f1_rf:.2f}")

# 训练逻辑回归模型
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 评估逻辑回归模型
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
print(f"逻辑回归 - 准确率: {accuracy_lr:.2f}, F1 分数: {f1_lr:.2f}")

# 比较模型表现
print("\n模型表现对比:")
print(f"决策树 - 准确率: {accuracy_dt:.2f}, F1 分数: {f1_dt:.2f}")
print(f"随机森林 - 准确率: {accuracy_rf:.2f}, F1 分数: {f1_rf:.2f}")
print(f"逻辑回归 - 准确率: {accuracy_lr:.2f}, F1 分数: {f1_lr:.2f}")

# 可视化模型表现
models = ['Decision Tree', 'Random Forest', 'Logistic Regression']
accuracies = [accuracy_dt, accuracy_rf, accuracy_lr]
f1_scores = [f1_dt, f1_rf, f1_lr]

plt.figure(figsize=(10, 6))

# 绘制准确率条形图
plt.subplot(1, 2, 1)
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# 绘制 F1 分数条形图
plt.subplot(1, 2, 2)
plt.bar(models, f1_scores, color=['blue', 'green', 'red'])
plt.title('Model F1 Score Comparison')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
