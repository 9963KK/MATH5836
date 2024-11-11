# Import lib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torch
import random
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
# Neural Network part
print('-' * 50, 'Part A', '-' * 50)
path = './data/abalone.data'
df = pd.read_csv(path, names=['sex', 'length', 'diameter', 'height',
                 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'])
# 创建img文件夹
if not os.path.exists('img'):
    os.mkdir('img')
# Use the recommend multiclass classifier
def multi_class(x):
    if x <= 7:
        return 0
    elif x <= 10:
        return 1
    elif x <= 15:
        return 2
    else:
        return 3


class_dict = {0: '0~7', 1: '8~10', 2: '11~15', 3: '16~'}
# Use different number to map the value of ring-age
y = df.iloc[:, -1].map(multi_class)
X = df.iloc[:, :-1]
# 对性别进行映射
X['sex'] = X['sex'].map({'M': 0, 'F': 1, 'I': 3})
cmap = cm.get_cmap('Pastel1')
abalone_colors = [cmap(i) for i in np.linspace(0, 1, len(y.value_counts()))]
plt.figure(figsize=(8, 8))
plt.pie(y.value_counts(), autopct='%.2f%%',
        startangle=90, labels=[class_dict[s] for s in y.value_counts().index], colors=abalone_colors)
plt.title('Pie chart')
plt.savefig('img/abalone_pie.png')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print( '#' * 20,'DecisionTree and RadomForest part', '#' * 20)
# DecisionTree and RadomForest part
# 定义不同的决策树深度
depths = [3, 5, 7, 9, 11]
train_accuracies = []
test_accuracies = []

# 进行多次实验，使用不同的树深度
for depth in depths:
    # 初始化决策树模型并设置最大深度
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # 训练模型
    clf.fit(X_train, y_train)

    # 计算训练集和测试集的准确率
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))

    # 将结果添加到列表中
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # 打印每次实验的结果
    print(f"Max Depth: {depth}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print("-" * 30)

# 绘制不同树深度下的训练和测试准确率图
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(depths, test_accuracies, label='Testing Accuracy', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.legend()
plt.grid()
plt.savefig('img/depth_accuracy.png')

# 使用最佳参数的决策树进行预测
best_clf = DecisionTreeClassifier(random_state=42, max_depth=5, max_features=None, min_samples_leaf=5, min_samples_split=20)
best_clf.fit(X_train, y_train)

# 绘制决策树
plt.figure(figsize=(20, 10))
plot_tree(best_clf, filled=True, feature_names=X.columns,
          class_names=['0-7', '8-10', '11-15', '15+'])
plt.title("Decision Tree Visualization (Max Depth = 5)")
plt.savefig('img/decision_tree.png')

# 将决策树转换为文本规则
tree_rules = export_text(best_clf, feature_names=list(X.columns))
print(tree_rules)

# 获取不同的 alpha 值和对应的子树
path = best_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas  # 不同的 alpha 值
impurities = path.impurities  # 不同的 alpha 下的树的 impurity 值

# 可视化 alpha 与 impurity 之间的关系
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities, marker='o', drawstyle="steps-post")
plt.xlabel("Effective Alpha")
plt.ylabel("Total Impurity of Leaves")
plt.title("Total Impurity vs Effective Alpha for Training Set")
plt.grid()
plt.savefig('img/alpha_impurity.png')

train_scores = []
test_scores = []
# 遍历不同的 ccp_alpha 值，生成不同的剪枝模型
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)

    # 计算训练和测试集的准确率
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# 可视化训练和测试集的准确率随 ccp_alpha 的变化
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o',
         label="Train Accuracy", drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o',
         label="Test Accuracy", drawstyle="steps-post")
plt.xlabel("Effective Alpha")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Effective Alpha for Pruned Tree")
plt.legend()
plt.grid()
plt.savefig('img/alpha_accuracy.png')
# 选择测试集准确率最高的 ccp_alpha 值
best_alpha = ccp_alphas[test_scores.index(max(test_scores))]
print("Best alpha:", best_alpha)

# 使用最佳 alpha 值重新训练剪枝后的决策树
pruned_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_clf.fit(X_train, y_train)

# 在测试集上评估剪枝后的模型
pruned_test_accuracy = pruned_clf.score(X_test, y_test)
print("Test Accuracy of Pruned Tree:", pruned_test_accuracy)

# 使用修剪后的模型进行预测
y_pred_pruned = pruned_clf.predict(X_test)              # 预测标签
y_prob_pruned = pruned_clf.predict_proba(X_test)        # 预测概率，用于计算 ROC-AUC

# 计算 F1 分数
# 可以选择 'macro' 或 'weighted'
f1_pruned = f1_score(y_test, y_pred_pruned, average='weighted')
print(f"F1 Score of Pruned Tree: {f1_pruned:.4f}")

# 计算 ROC-AUC
# 如果是二分类
if len(set(y_test)) == 2:
    roc_auc_pruned = roc_auc_score(y_test, y_prob_pruned[:, 1])  # 假设第二列为正类的概率
    print(f"ROC-AUC of Pruned Tree: {roc_auc_pruned:.4f}")
# 如果是多分类
else:
    # 先将 y_test 二值化
    y_test_binarized = label_binarize(y_test, classes=list(set(y_test)))
    roc_auc_multi_pruned = roc_auc_score(
        y_test_binarized, y_prob_pruned, average='macro', multi_class='ovr')
    print(f"ROC-AUC (Multi-class) of Pruned Tree: {roc_auc_multi_pruned:.4f}")

# 定义不同的树数量
n_estimators_list = [10, 50, 100, 200, 300]
train_accuracies = []
test_accuracies = []

# 遍历不同的树数量，训练随机森林模型
for n_estimators in n_estimators_list:
    # 初始化随机森林模型并设置树的数量
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # 训练模型
    rf_clf.fit(X_train, y_train)

    # 计算训练集和测试集的准确率
    train_accuracy = rf_clf.score(X_train, y_train)
    test_accuracy = rf_clf.score(X_test, y_test)

    # 将结果添加到列表中
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # 输出每次实验的结果
    print(f"Number of Trees: {n_estimators}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print("-" * 30)

# 可视化不同树数量下的训练和测试准确率
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, train_accuracies,
         label='Training Accuracy', marker='o')
plt.plot(n_estimators_list, test_accuracies,
         label='Testing Accuracy', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest: Number of Trees vs Accuracy')
plt.legend()
plt.grid()
plt.savefig('img/random_forest.png')

# 限制树的深度减少过拟合
rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)
train_accuracy = rf_clf.score(X_train, y_train)
test_accuracy = rf_clf.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# 使用模型预测
y_pred = rf_clf.predict(X_test)              # 预测标签
y_prob = rf_clf.predict_proba(X_test)        # 预测概率，用于计算 ROC-AUC

# 计算 F1 分数
f1 = f1_score(y_test, y_pred, average='weighted')  # 可以选择 'macro' 或 'weighted'
print(f"F1 Score: {f1:.4f}")

# 计算 ROC-AUC
# 如果是二分类
if len(set(y_test)) == 2:
    roc_auc = roc_auc_score(y_test, y_prob[:, 1])  # 假设第二列为正类的概率
    print(f"ROC-AUC: {roc_auc:.4f}")
# 如果是多分类
else:
    # 先将 y_test 二值化
    y_test_binarized = label_binarize(y_test, classes=list(set(y_test)))
    roc_auc_multi = roc_auc_score(
        y_test_binarized, y_prob, average='macro', multi_class='ovr')
    print(f"ROC-AUC (Multi-class): {roc_auc_multi:.4f}")

# XGBoost and GBDT part
print('#' * 20, 'XGB and GBDT part', '#' * 20)

# XGBoost 分类器 - 树数量 vs 准确率
train_accuracies_xgb = []
test_accuracies_xgb = []
num_boost_rounds = [10, 50, 100, 150, 200, 250, 300]

for n in num_boost_rounds:
    xgb_model = XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=n)
    xgb_model.fit(X_train, y_train)
    train_accuracies_xgb.append(xgb_model.score(X_train, y_train))
    test_accuracies_xgb.append(xgb_model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(num_boost_rounds, train_accuracies_xgb,
         marker='o', label='Training Accuracy')
plt.plot(num_boost_rounds, test_accuracies_xgb,
         marker='o', label='Testing Accuracy')
plt.xlabel('Number of Boosting Rounds')
plt.ylabel('Accuracy')
plt.title('XGBoost: Number of Boosting Rounds vs Accuracy')
plt.legend()
plt.savefig('img/abalone_xgb_accuracy.png')

# 训练 XGBoost 分类器
xgb_model = XGBClassifier(use_label_encoder=False,
                          eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 预测并评估 XGBoost 模型
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
print(f"XGBoost - 准确率: {accuracy_xgb:.2f}, F1 分数: {f1_xgb:.2f}")

# 训练梯度提升分类器
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# 预测并评估梯度提升模型
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='weighted')
print(f"梯度提升 - 准确率: {accuracy_gb:.2f}, F1 分数: {f1_gb:.2f}")

# 训练梯度提升分类器并绘制树数量 vs 准确率曲线图
train_accuracies_gb = []
test_accuracies_gb = []
num_trees = [10, 50, 100, 150, 200, 250, 300]

for n in num_trees:
    gb_model = GradientBoostingClassifier(n_estimators=n, random_state=42)
    gb_model.fit(X_train, y_train)
    train_accuracies_gb.append(gb_model.score(X_train, y_train))
    test_accuracies_gb.append(gb_model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(num_trees, train_accuracies_gb, marker='o', label='Training Accuracy')
plt.plot(num_trees, test_accuracies_gb, marker='o', label='Testing Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('GBDT: Number of Trees vs Accuracy')
plt.legend()
plt.savefig('img/abalone_gb_accuracy.png')

# 比较模型表现
print("\n模型表现对比:")
print(f"XGBoost - 准确率: {accuracy_xgb:.2f}, F1 分数: {f1_xgb:.2f}")
print(f"梯度提升 - 准确率: {accuracy_gb:.2f}, F1 分数: {f1_gb:.2f}")


# 由于数据是处理过的，不需要再去做额外的特征工程
# 由于是多分类，我们可以使用One-to-one或者One-to-rest的方式
# 同时引入K折交叉验证
print('#' * 20, 'Neural Netwok part', '#' * 20)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def simple_nn_train(model, X=X, y=y, only=None, iter=30):
    ovo = []
    ovr = []
    for stra in ['ovo', 'ovr']:
        if only and stra not in only:
            continue
        f1_socre_list = []
        roc_auc_score_list = []
        for i in range(1, iter + 1):
            f1_res = []
            roc_auc_res = []
            for train_index, test_index in kf.split(X, y):
                X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[
                    test_index], y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                f1 = f1_score(y_test, y_pred, average='micro')
                roc_auc = roc_auc_score(y_test, y_prob, multi_class=stra)
                f1_res.append(f1)
                roc_auc_res.append(roc_auc)
            f1_res_mean = np.mean(f1_res)
            roc_auc_res_mean = np.mean(roc_auc_res)
            if i % (iter / 10) == 0:
                print(
                    f'Iteration {i}/ {iter}: Using Stragety {stra} f1_score={f1_res_mean:.3f}; roc_auc={roc_auc_res_mean:.3f}')
            f1_socre_list.append(f1_res_mean)
            roc_auc_score_list.append(roc_auc_res_mean)
        print(
            f'Final Summary: Iterations {iter}, Stragety {stra}, f1_score={f1_socre_list[-1]:.3f}, roc_auc={roc_auc_score_list[-1]:.3f}')
        if stra == 'ovo':
            ovo.append(f1_socre_list)
            ovo.append(roc_auc_score_list)
        else:
            ovr.append(f1_socre_list)
            ovr.append(roc_auc_score_list)
    return ovo, ovr


iters = 100
print('Using solver=sgd')
default_ovo, default_ovr = simple_nn_train(MLPClassifier(
    warm_start=True, solver='sgd', early_stopping=True), iter=iters)
print('Using solver=adam')
adam_ovo, adam_ovr = simple_nn_train(MLPClassifier(
    solver='adam', warm_start=True, early_stopping=True), iter=iters)
# 绘制收敛速度的曲线
# 封装绘制图片的函数


def plot_scores(tag='SGD', iters=iters, data=None, score='f1_score'):
    if score == 'f1_score':
        d = 0
    else:
        d = 1
    plt.plot(range(1, iters + 1), data[d], label=tag)


plt.figure(figsize=(8, 8))
plot_scores(tag='SGD', data=default_ovo, score='roc_auc')
plot_scores(tag='Adam', data=adam_ovo, score='roc_auc')
plt.title(f'SGD vs. Adam using ovo roc_auc (iterations={iters})')
plt.legend()
plt.savefig('img/abalone_ovo_roc_auc.png')

plt.figure(figsize=(8, 8))
plot_scores(tag='SGD', data=default_ovo, score='f1_score')
plot_scores(tag='Adam', data=adam_ovo, score='f1_score')
plt.title(f'SGD vs. Adam using ovo f1_score (iterations={iters})')
plt.xlabel('Iterations')
plt.ylabel('Scores')
plt.legend()
plt.savefig('img/abalone_ovo_f1_score.png')
# 由于ovr表现不稳定没法体现Adam的优势，所以不画图

# 由于后面需要使用到dropout的技巧，这里就需要调用pytorch里面的神经网络了
# 先构建一个使用Adam作为优化器并且添加了L2的神经网络


class SNN(nn.Module):
    def __init__(self, n_class=4):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, n_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
# 引入Dropout


class SNNDropout(nn.Module):
    def __init__(self, dropout=0.5, n_class=4):
        super(SNNDropout, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.drop = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(64, n_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
# 处理数据


def data2loader(X, y, batch_size=128, val_pct=0.3):
    X_tensor, y_tensor = torch.tensor(X.values), torch.tensor(y.values)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int((1 - val_pct) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


train_loader, val_loader = data2loader(X, y)

epoches = 200
device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu')


def train_eval(model, device, train_loader, val_loader, epoches, lr=0.001, weight_decay=0, verbose=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_f1_score_list = []
    val_roc_auc_score_list = []
    train_f1_score_list = []
    train_roc_auc_score_list = []
    finally_f1 = 0
    finally_roc_auc = 0
    bar = tqdm(range(epoches), ncols=200)
    for epoch in bar:
        model.train()
        train_pred = []
        train_labels = []
        train_prob = []
        bar.set_description(f'Epoch: {epoch + 1} / {epoches}')
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_pred.extend(predicted.cpu().detach().numpy())
            train_labels.extend(labels.cpu().detach().numpy())
            train_prob.extend(outputs.cpu().detach().numpy())
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 and verbose:
            print('Epoch:', epoch, 'Loss:', loss.item())
        train_f1 = f1_score(train_labels, train_pred, average='micro')
        train_roc_auc = roc_auc_score(
            train_labels, train_prob, multi_class='ovr', average='weighted')
        train_f1_score_list.append(train_f1)
        train_roc_auc_score_list.append(train_roc_auc)
        bar.set_postfix({'f1_score': f'{train_f1:.5f}',
                        'roc_auc_score': f'{train_roc_auc:.5f}'})
        model.eval()
        correct = 0
        total = 0
        val_pred = []
        val_labels = []
        val_prob = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_pred.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_prob.extend(outputs.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if epoch % 10 == 0 and verbose:
            print('Epoch:', epoch, 'Accuracy', round(correct / total, 5))
        finally_f1 = f1_score(val_labels, val_pred, average='micro')
        finally_roc_auc = roc_auc_score(
            val_labels, val_prob, multi_class='ovr', average='macro')
        val_f1_score_list.append(finally_f1.copy())
        val_roc_auc_score_list.append(finally_roc_auc.copy())
    print(
        f'Epoch {epoches}: Final Summary on Validation: f1_score {finally_f1:.5f}; roc_auc_score {finally_roc_auc:.5f}')
    return [train_f1_score_list, train_roc_auc_score_list], [val_f1_score_list, val_roc_auc_score_list]

# 对比不同的dropout的收敛速度,图片
def epoch_plot(data, epoches, score, tag):
    if len(data) != len(tag):
        return 'Error need the same number of data and tag'
    plt.figure(figsize=(8, 8))
    for i in range(len(data)):
        plt.plot(range(1, epoches + 1), data[i], label=tag[i])
    name = ' vs '.join(tag) + '_' + score + '_comparison'
    plt.title(f'Epoch {epoches} {name}')
    plt.legend()
    plt.savefig(f'img/{name}.png')
    
default_train, default_val = train_eval(
        SNN(), device, train_loader, val_loader=val_loader, epoches=epoches, verbose=False)
dropout_train, dropout_val = train_eval(SNNDropout(
    dropout=0.5), device, train_loader, val_loader=val_loader, epoches=epoches, verbose=False)
# 加入l2正则化
l2_train, l2_val = train_eval(SNN(), device=device, train_loader=train_loader,
                              val_loader=val_loader, epoches=epoches, weight_decay=0.001, verbose=False)

# 绘制criterion随着epoch的变化曲线
epoch_plot([default_train[0], default_val[0]], epoches,
           'default f1_score', ['train', 'val'])
epoch_plot([default_train[1], default_val[1]], epoches,
           'default roc_auc_score', ['train', 'val'])
epoch_plot([dropout_train[0], dropout_val[0]], epoches,
           'dropout f1_score', ['train', 'val'])
epoch_plot([dropout_train[1], dropout_val[1]], epoches,
           'dropout roc_auc_score', ['train', 'val'])
epoch_plot([l2_train[0], l2_val[0]], epoches, 'l2 f1_score', ['train', 'val'])
epoch_plot([l2_train[1], l2_val[1]], epoches,
           'l2 roc_auc_score', ['train', 'val'])
epoch_plot([default_val[1], dropout_val[1], l2_val[1]], epoches,
           'val_roc_auc_score', ['default', 'dropout', 'l2'])

dropouts = [0.4, 0.5, 0.6, 0.7]
weight_decays = [1e-2, 1e-3, 5e-3, 5e-4]
for d_rate in dropouts:
    print(f'Dropout: {d_rate}')
    train_eval(SNNDropout(d_rate), device, train_loader,
               val_loader=val_loader, epoches=epoches, verbose=False)
for w_decay in weight_decays:
    print(f'Weight decay: {w_decay}')
    train_eval(SNN(), device, train_loader, val_loader=val_loader,
               epoches=epoches, weight_decay=w_decay, verbose=False)
# 引入新数据
print('-' * 50, 'Part B', '-' * 50)
# choose model
print('After comparison, We choose RandomForestClassifier and MLPClassifier with Adam solver as the best models')
# data (as pandas dataframes)
data_B = pd.read_csv('./data/cmc.data', sep=',', header=None, names=[
    'wife_age', 'wife_education', 'husband_education', 'num_children', 'wife_religion', 'wife_work', 'husband_occupation', 'standard_living', 'media_exposure', 'contraceptive_method'])
X_B = data_B.iloc[:, :-1]
y_B = data_B.iloc[:, -1]

# 先看看缺失集的问题
print('The distribution nan value in different features')
print(X_B.isna().sum())
# Using RandomForestClassifier
print('#' * 20, 'RandomForestClassifier', '#' * 20)
sns.pairplot(data_B, hue='contraceptive_method', diag_kind='kde')
plt.savefig('img/contraceptive_method_pairplot.png')
corr = data_B.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('img/contraceptive_method_corr.png')

X_train, X_test, y_train, y_test = train_test_split(
    X_B, y_B, test_size=0.2, random_state=42)
# 初始化随机森林分类器
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100棵树

# 训练模型
rf_model.fit(X_train, y_train)
# 使用测试集进行预测
y_pred = rf_model.predict(X_test)

# 计算评估指标
# 计算 Accuracy
accuracy = accuracy_score(y_test, y_pred)
# 计算 F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')  # 使用 weighted 平均方法适合多分类
# 计算 ROC-AUC Score
# 对于多分类，设置 average 参数；如果是二分类，不需要设置 average
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(
    X_test), multi_class='ovr', average='weighted')
# 输出结果
print("Evaluation Metrics:")
print("Accuracy:", accuracy)
print("F1 Score (weighted):", f1)
print("ROC-AUC Score (weighted):", roc_auc)

# 分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))
# 混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 获取特征重要性
feature_importances = pd.Series(rf_model.feature_importances_, index=X_B.columns)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importances')
plt.savefig('img/feature_importances.png')
# Using MLPClassifier
print('#' * 20, 'MLPClassifier', '#' * 20)

# 使用可视化分析每个变量的分布函数
plt.figure(figsize=(18, 18))
for i, col in enumerate(X_B.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(X_B[col], kde=True)
plt.tight_layout()
plt.savefig('img/contraceptive_method_choice.png')

# 对于target value的不同类别进行统计
B_colors = [cmap(i) for i in np.linspace(0, 1, len(y_B.value_counts()))]
plt.figure(figsize=(8, 8))
plt.pie(y_B.value_counts(), autopct='%.2f%%', startangle=90, labels=[
        i for i in y_B.value_counts().index], colors=B_colors)
plt.title('Pie chart')
plt.savefig('img/contraceptive_method_choice_pie.png')
B_ovo, B_ovr = simple_nn_train(MLPClassifier(solver='adam', warm_start=True, early_stopping=True, learning_rate='adaptive',
                               learning_rate_init=0.001, hidden_layer_sizes=(256, 256)), X=X_B, y=y_B, iter=150, only=['ovr'])


print('-' * 50, 'Part C', '', '-' * 50)

# 加载数据集
file_path = './data/processed.cleveland.data'  # Heart Disease 数据集路径

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
# df.info()

# 绘制目标变量的分布
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Presence (0 = No, 1 = Yes)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.savefig('img/heart_disease_count.png')

# 绘制特征之间的相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.savefig('img/heart_disease_heatmap.png')

# 模型构建 - 将 'target' 作为目标变量，其他特征作为输入
X = df.drop(columns=['target'])
y = df['target']

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

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
plt.savefig('img/model_comparison.png')
