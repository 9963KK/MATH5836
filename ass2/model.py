# Neural Network part
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings

# Neural Network part
warnings.filterwarnings("ignore")
path = '../abalone/abalone.data'
df = pd.read_csv(path, names=['sex', 'length', 'diameter', 'height',
                 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'])

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
# 这里用的是老师的方法分类,并不均匀
cmap = cm.get_cmap('Pastel1')
abalone_colors = [cmap(i) for i in np.linspace(0, 1, len(y.value_counts()))]
plt.figure(figsize=(8, 8))
plt.pie(y.value_counts(), autopct='%.2f%%',
        startangle=90, labels=[class_dict[s] for s in y.value_counts().index], colors=abalone_colors)
plt.title('Pie chart')
plt.savefig('abalone_pie.png')

# 由于数据是处理过的，不需要再去做额外的特征工程
# 由于是多分类，我们可以使用One-to-one或者One-to-rest的方式
# 同时引入K折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def simple_nn_train(model, X=X, y=y, iter=30):
    ovo = []
    ovr = []
    for stra in ['ovo', 'ovr']:
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
            # if i % (iter / 10) == 0:
            #     print(
            #         f'Iteration {i}/ {iter}: Using Stragety {stra} f1_score={f1_res_mean:.3f}; roc_auc={roc_auc_res_mean:.3f}')
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
print('Default Configuration')
default_ovo, default_ovr = simple_nn_train(MLPClassifier(
    warm_start=True, early_stopping=True), iter=iters)
print('Using solver=adam')
adam_ovo, adam_ovr = simple_nn_train(MLPClassifier(
    solver='adam', warm_start=True, early_stopping=True), iter=iters)

# 绘制收敛速度的曲线
# 封装绘制图片的函数


def plot_scores(tag='Default', iters=iters, data=None, score='f1_score'):
    if score == 'f1_score':
        d = 0
    else:
        d = 1
    plt.plot(range(1, iters + 1), data[d], label=tag)


plt.figure(figsize=(8, 8))
plot_scores(tag='Default', data=default_ovo, score='roc_auc')
plot_scores(tag='Adam', data=adam_ovo, score='roc_auc')
plt.title(f'Default vs. Adam using ovo roc_auc (iterations={iters})')
plt.legend()
plt.savefig('abalone_ovo_roc_auc.png')

plt.figure(figsize=(8, 8))
plot_scores(tag='Default', data=default_ovo, score='f1_score')
plot_scores(tag='Adam', data=adam_ovo, score='f1_score')
plt.title(f'Default vs. Adam using ovo f1_score (iterations={iters})')
plt.xlabel('Iterations')
plt.ylabel('Scores')
plt.legend()
plt.savefig('abalone_ovo_f1_score.png')
# 由于ovr表现不稳定没法体现Adam的优势，所以不画图

# 由于后面需要使用到dropout的技巧，这里就需要调用pytorch里面的神经网络了
# 先构建一个使用Adam作为优化器并且添加了L2的神经网络


class SGD(nn.Module):
    def __init__(self, n_class=4):
        super(SGD, self).__init__()
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


class SGDDropout(nn.Module):
    def __init__(self, dropout=0.5, n_class=4):
        super(SGDDropout, self).__init__()
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


def data2loader(X, y, batch_size=64, val_pct=0.3):
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

epoches = 100
device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu')


def train_eval(model, device, train_loader, val_loader, epoches, lr=0.001, weight_decay=0, verbose=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    finally_f1 = 0
    finally_roc_auc = 0
    for epoch in range(epoches):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            labels_cpu = labels.cpu().numpy()
            loss.backward()
            optimizer.step()
        # if epoch % 10 == 0 and verbose:
        #     print('Epoch:', epoch, 'Loss:', loss.item())
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
        # if epoch % 10 == 0 and verbose:
        #     print('Epoch:', epoch, 'Accuracy', round(correct / total, 5))
        finally_f1 = f1_score(val_labels, val_pred, average='micro')
        finally_roc_auc = roc_auc_score(
            val_labels, val_prob, multi_class='ovo', average='macro')
    print(
        f'Final Summary: Epoch {epoches} f1_score {finally_f1:.5f}; roc_auc_score {finally_roc_auc:.5f}')


# train_eval(SGD(), device, train_loader, val_loader=val_loader, epoches=epoches)
# train_eval(SGDDropout(), device, train_loader,
#            val_loader=val_loader, epoches=epoches)
# # 加入l2正则化
# train_eval(SGD(), device=device, train_loader=train_loader,
#            val_loader=val_loader, epoches=100, weight_decay=0.001)
# 尝试不同的combination来测试
dropouts = [0.4, 0.5, 0.6, 0.7]
weight_decays = [1e-2, 1e-3, 5e-3, 5e-4]
for d_rate in dropouts:
    print(f'Dropout: {d_rate}')
    train_eval(SGDDropout(d_rate), device, train_loader,
               val_loader=val_loader, epoches=epoches, verbose=False)
for w_decay in weight_decays:
    print(f'Weight decay: {w_decay}')
    train_eval(SGD(), device, train_loader, val_loader=val_loader,
               epoches=epoches, weight_decay=w_decay, verbose=False)
# 引入新数据
print('-' * 100)
# fetch dataset

# data (as pandas dataframes)
data_B = pd.read_csv('../cmc/contraceptive_method_choice.data', sep=',', header=None, names=[
    'wife_age', 'wife_education', 'husband_education', 'num_children', 'wife_religion', 'wife_work', 'husband_occupation', 'standard_living', 'media_exposure', 'contraceptive_method'])
X_B = data_B.iloc[:, :-1]
y_B = data_B.iloc[:, -1]

# 先看看缺失集的问题
print('The distribution nan value in different features')
X_B.isna().sum()

# 使用可视化分析每个变量的分布函数
plt.figure(figsize=(18, 18))
for i, col in enumerate(X_B.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(X_B[col], kde=True)
plt.tight_layout()
plt.savefig('contraceptive_method_choice.png')

# 对于target value的不同类别进行统计
B_colors = [cmap(i) for i in np.linspace(0, 1, len(y_B.value_counts()))]
plt.figure(figsize=(8, 8))
plt.pie(y_B.value_counts(), autopct='%.2f%%', startangle=90, labels=[
        i for i in y_B.value_counts().index], colors=B_colors)
plt.title('Pie chart')
plt.savefig('contraceptive_method_choice_pie.png')
B_ovo, B_ovr = simple_nn_train(MLPClassifier(
    solver='adam', warm_start=True, early_stopping=True), X=X_B, y=y_B, iter=100)
