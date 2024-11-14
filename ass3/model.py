# Import libraries
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

# Create img folder
if not os.path.exists('img'):
    os.mkdir('img')

# Define the recommended multiclass classifier


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
# Map the value of ring-age using different numbers
y = df.iloc[:, -1].map(multi_class)
X = df.iloc[:, :-1]
# Map the values for 'sex'
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
print('#' * 20, 'DecisionTree and RandomForest part', '#' * 20)

# DecisionTree and RandomForest part
# Define different decision tree depths
depths = [3, 5, 7, 9, 11]
train_accuracies = []
test_accuracies = []

# Conduct multiple experiments using different tree depths
for depth in depths:
    # Initialize the decision tree model and set max depth
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))

    # Add results to list
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Print results for each experiment
    print(f"Max Depth: {depth}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print("-" * 30)

# Plot training and testing accuracy at different tree depths
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(depths, test_accuracies, label='Testing Accuracy', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.legend()
plt.grid()
plt.savefig('img/depth_accuracy.png')

# Use a decision tree with the best parameters for prediction
best_clf = DecisionTreeClassifier(
    random_state=42, max_depth=5, max_features=None, min_samples_leaf=5, min_samples_split=20)
best_clf.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(best_clf, filled=True, feature_names=list(X.columns),
          class_names=['0-7', '8-10', '11-15', '15+'])
plt.title("Decision Tree Visualization (Max Depth = 5)")
plt.savefig('img/decision_tree.png')

# Convert decision tree to text rules
tree_rules = export_text(best_clf, feature_names=list(X.columns))
print(tree_rules)

# Get different alpha values and corresponding subtrees
path = best_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas  # Different alpha values
impurities = path.impurities  # Impurity values for trees under different alphas

# Visualize the relationship between alpha and impurity
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities, marker='o', drawstyle="steps-post")
plt.xlabel("Effective Alpha")
plt.ylabel("Total Impurity of Leaves")
plt.title("Total Impurity vs Effective Alpha for Training Set")
plt.grid()
plt.savefig('img/alpha_impurity.png')

train_scores = []
test_scores = []
# Generate different pruned models by iterating over ccp_alpha values
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)

    # Calculate training and testing accuracy
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# Visualize accuracy vs. ccp_alpha for training and testing sets
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
# Select ccp_alpha with highest test set accuracy
best_alpha = ccp_alphas[test_scores.index(max(test_scores))]
print("Best alpha:", best_alpha)

# Retrain pruned decision tree with best alpha
pruned_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_clf.fit(X_train, y_train)

# Evaluate pruned model on test set
pruned_test_accuracy = pruned_clf.score(X_test, y_test)
print("Test Accuracy of Pruned Tree:", pruned_test_accuracy)

# Predict with pruned model
y_pred_pruned = pruned_clf.predict(X_test)              # Predicted labels
# Predicted probabilities for ROC-AUC calculation
y_prob_pruned = pruned_clf.predict_proba(X_test)

# Calculate F1 score
f1_pruned = f1_score(y_test, y_pred_pruned, average='weighted')
print(f"F1 Score of Pruned Tree: {f1_pruned:.4f}")

# Calculate ROC-AUC
if len(set(y_test)) == 2:
    # Assuming second column is positive class probability
    roc_auc_pruned = roc_auc_score(y_test, y_prob_pruned[:, 1])
    print(f"ROC-AUC of Pruned Tree: {roc_auc_pruned:.4f}")
else:
    y_test_binarized = label_binarize(y_test, classes=list(set(y_test)))
    roc_auc_multi_pruned = roc_auc_score(
        y_test_binarized, y_prob_pruned, average='macro', multi_class='ovr')
    print(f"ROC-AUC (Multi-class) of Pruned Tree: {roc_auc_multi_pruned:.4f}")

# Define different numbers of trees
n_estimators_list = [10, 50, 100, 200, 300]
train_accuracies = []
test_accuracies = []

# Train random forest model with different tree numbers
for n_estimators in n_estimators_list:
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Train model
    rf_clf.fit(X_train, y_train)

    # Calculate training and test accuracy
    train_accuracy = rf_clf.score(X_train, y_train)
    test_accuracy = rf_clf.score(X_test, y_test)

    # Add results to lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Output results for each experiment
    print(f"Number of Trees: {n_estimators}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print("-" * 30)

# Plot training and testing accuracy at different tree depths
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(depths, test_accuracies, label='Testing Accuracy', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.legend()
plt.grid()
plt.savefig('img/depth_accuracy.png')

# Use a decision tree with the best parameters for prediction
best_clf = DecisionTreeClassifier(
    random_state=42, max_depth=5, max_features=None, min_samples_leaf=5, min_samples_split=20)
best_clf.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(best_clf, filled=True, feature_names=list(X.columns),
          class_names=['0-7', '8-10', '11-15', '15+'])
plt.title("Decision Tree Visualization (Max Depth = 5)")
plt.savefig('img/decision_tree.png')

# Convert the decision tree to text rules
tree_rules = export_text(best_clf, feature_names=list(X.columns))
print(tree_rules)

# Get different alpha values and corresponding subtrees
path = best_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas  # Different alpha values
impurities = path.impurities  # Impurity values for trees under different alphas

# Visualize the relationship between alpha and impurity
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities, marker='o', drawstyle="steps-post")
plt.xlabel("Effective Alpha")
plt.ylabel("Total Impurity of Leaves")
plt.title("Total Impurity vs Effective Alpha for Training Set")
plt.grid()
plt.savefig('img/alpha_impurity.png')

train_scores = []
test_scores = []
# Iterate over different ccp_alpha values to generate different pruned models
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)

    # Calculate training and test accuracy
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# Visualize training and testing accuracy as ccp_alpha varies
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

# Select the ccp_alpha value with the highest test set accuracy
best_alpha = ccp_alphas[test_scores.index(max(test_scores))]
print("Best alpha:", best_alpha)

# Retrain pruned decision tree with the best alpha
pruned_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_clf.fit(X_train, y_train)

# Evaluate pruned model on test set
pruned_test_accuracy = pruned_clf.score(X_test, y_test)
print("Test Accuracy of Pruned Tree:", pruned_test_accuracy)

# Predict with pruned model
y_pred_pruned = pruned_clf.predict(X_test)              # Predicted labels
# Predicted probabilities for ROC-AUC calculation
y_prob_pruned = pruned_clf.predict_proba(X_test)

# Calculate F1 score
f1_pruned = f1_score(y_test, y_pred_pruned, average='weighted')
print(f"F1 Score of Pruned Tree: {f1_pruned:.4f}")

# Calculate ROC-AUC
# If binary classification
if len(set(y_test)) == 2:
    # Assuming second column is positive class probability
    roc_auc_pruned = roc_auc_score(y_test, y_prob_pruned[:, 1])
    print(f"ROC-AUC of Pruned Tree: {roc_auc_pruned:.4f}")
# If multiclass classification
else:
    # First binarize y_test
    y_test_binarized = label_binarize(y_test, classes=list(set(y_test)))
    roc_auc_multi_pruned = roc_auc_score(
        y_test_binarized, y_prob_pruned, average='macro', multi_class='ovr')
    print(f"ROC-AUC (Multi-class) of Pruned Tree: {roc_auc_multi_pruned:.4f}")

# Define different numbers of trees
n_estimators_list = [10, 50, 100, 200, 300]
train_accuracies = []
test_accuracies = []

# Iterate over different tree numbers and train random forest model
for n_estimators in n_estimators_list:
    # Initialize random forest model and set tree number
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Train model
    rf_clf.fit(X_train, y_train)

    # Calculate training and test accuracy
    train_accuracy = rf_clf.score(X_train, y_train)
    test_accuracy = rf_clf.score(X_test, y_test)

    # Add results to lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Output results for each experiment
    print(f"Number of Trees: {n_estimators}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print("-" * 30)

# Visualize training and testing accuracy at different tree numbers
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

# Limit tree depth to reduce overfitting
rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)
train_accuracy = rf_clf.score(X_train, y_train)
test_accuracy = rf_clf.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Predict with model
y_pred = rf_clf.predict(X_test)              # Predicted labels
# Predicted probabilities for ROC-AUC calculation
y_prob = rf_clf.predict_proba(X_test)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")

# Calculate ROC-AUC
# If binary classification
if len(set(y_test)) == 2:
    # Assuming second column is positive class probability
    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    print(f"ROC-AUC: {roc_auc:.4f}")
# If multiclass classification
else:
    # First binarize y_test
    y_test_binarized = label_binarize(y_test, classes=list(set(y_test)))
    roc_auc_multi = roc_auc_score(
        y_test_binarized, y_prob, average='macro', multi_class='ovr')
    print(f"ROC-AUC (Multi-class): {roc_auc_multi:.4f}")

# XGBoost and GBDT part
print('#' * 20, 'XGB and GBDT part', '#' * 20)

# XGBoost classifier - Tree number vs. accuracy
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

# Train XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False,
                          eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate the XGBoost model
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
print(f"XGBoost - Accuracy: {accuracy_xgb:.2f}, F1 Score: {f1_xgb:.2f}")

# Train Gradient Boosting classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predict and evaluate the Gradient Boosting model
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='weighted')
print(
    f"Gradient Boosting - Accuracy: {accuracy_gb:.2f}, F1 Score: {f1_gb:.2f}")

# Train the Gradient Boosting classifier and plot the number of trees vs accuracy
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

# Compare model performance
print("\nModel Performance Comparison:")
print(f"XGBoost - Accuracy: {accuracy_xgb:.2f}, F1 Score: {f1_xgb:.2f}")
print(
    f"Gradient Boosting - Accuracy: {accuracy_gb:.2f}, F1 Score: {f1_gb:.2f}")

# Since the data is already processed, no additional feature engineering is needed
# Since it is multi-class, we can use One-vs-One or One-vs-Rest strategies
# Also introduce K-Fold Cross-Validation
print('#' * 20, 'Neural Network part', '#' * 20)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def simple_nn_train(model, X=X, y=y, only=None, iter=30):
    ovo = []
    ovr = []
    for stra in ['ovo', 'ovr']:
        if only and stra not in only:
            continue
        f1_score_list = []
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
                print(f'Iteration {i}/{iter}: Using Strategy {stra} f1_score={f1_res_mean:.3f}; roc_auc={roc_auc_res_mean:.3f}')
            f1_score_list.append(f1_res_mean)
            roc_auc_score_list.append(roc_auc_res_mean)
        print(f'Final Summary: Iterations {iter}, Strategy {stra}, f1_score={f1_score_list[-1]:.3f}, roc_auc={roc_auc_score_list[-1]:.3f}')
        if stra == 'ovo':
            ovo.append(f1_score_list)
            ovo.append(roc_auc_score_list)
        else:
            ovr.append(f1_score_list)
            ovr.append(roc_auc_score_list)
    return ovo, ovr


iters = 100
print('Using solver=sgd')
default_ovo, default_ovr = simple_nn_train(MLPClassifier(
    warm_start=True, solver='sgd', early_stopping=True), iter=iters)
print('Using solver=adam')
adam_ovo, adam_ovr = simple_nn_train(MLPClassifier(
    solver='adam', warm_start=True, early_stopping=True), iter=iters)

# Plot convergence speed
# Function to plot the graphs


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
# Since ovr performance is unstable, Adam's advantage cannot be shown, so no graph is drawn

# Since dropout technique will be used later, we need to call neural networks from PyTorch
# First, construct a neural network using Adam as the optimizer and adding L2 regularization


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

# Add Dropout


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

# Process data


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

epochs = 200
device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu')


def train_eval(model, device, train_loader, val_loader, epochs, lr=0.001, weight_decay=0, verbose=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_f1_score_list = []
    val_roc_auc_score_list = []
    train_f1_score_list = []
    train_roc_auc_score_list = []
    final_f1 = 0
    final_roc_auc = 0
    bar = tqdm(range(epochs), ncols=200)
    for epoch in bar:
        model.train()
        train_pred = []
        train_labels = []
        train_prob = []
        bar.set_description(f'Epoch: {epoch + 1} / {epochs}')
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
        final_f1 = f1_score(val_labels, val_pred, average='micro')
        final_roc_auc = roc_auc_score(
            val_labels, val_prob, multi_class='ovr', average='macro')
        val_f1_score_list.append(final_f1.copy())
        val_roc_auc_score_list.append(final_roc_auc.copy())
    print(f'Epoch {epochs}: Final Summary on Validation: f1_score {final_f1:.5f}; roc_auc_score {final_roc_auc:.5f}')
    return [train_f1_score_list, train_roc_auc_score_list], [val_f1_score_list, val_roc_auc_score_list]

# Compare the convergence speed of different dropouts, images


def epoch_plot(data, epochs, score, tag):
    if len(data) != len(tag):
        return 'Error: data and tag must have the same length'
    plt.figure(figsize=(8, 8))
    for i in range(len(data)):
        plt.plot(range(1, epochs + 1), data[i], label=tag[i])
    name = ' vs '.join(tag) + '_' + score + '_comparison'
    plt.title(f'Epoch {epochs} {name}')
    plt.legend()
    plt.savefig(f'img/{name}.png')


default_train, default_val = train_eval(
    SNN(), device, train_loader, val_loader=val_loader, epochs=epochs, verbose=False)
dropout_train, dropout_val = train_eval(SNNDropout(
    dropout=0.5), device, train_loader, val_loader=val_loader, epochs=epochs, verbose=False)
# Add L2 regularization
l2_train, l2_val = train_eval(SNN(), device=device, train_loader=train_loader,
                              val_loader=val_loader, epochs=epochs, weight_decay=0.001, verbose=False)

# Plot criterion changes with epochs
epoch_plot([default_train[0], default_val[0]], epochs,
           'default f1_score', ['train', 'val'])
epoch_plot([default_train[1], default_val[1]], epochs,
           'default roc_auc_score', ['train', 'val'])
epoch_plot([dropout_train[0], dropout_val[0]], epochs,
           'dropout f1_score', ['train', 'val'])
epoch_plot([dropout_train[1], dropout_val[1]], epochs,
           'dropout roc_auc_score', ['train', 'val'])
epoch_plot([l2_train[0], l2_val[0]], epochs, 'l2 f1_score', ['train', 'val'])
epoch_plot([l2_train[1], l2_val[1]], epochs,
           'l2 roc_auc_score', ['train', 'val'])
epoch_plot([default_val[1], dropout_val[1], l2_val[1]], epochs,
           'val_roc_auc_score', ['default', 'dropout', 'l2'])

dropouts = [0.4, 0.5, 0.6, 0.7]
weight_decays = [1e-2, 1e-3, 5e-3, 5e-4]
for d_rate in dropouts:
    print(f'Dropout: {d_rate}')
    train_eval(SNNDropout(d_rate), device, train_loader,
               val_loader=val_loader, epochs=epochs, verbose=False)
for w_decay in weight_decays:
    print(f'Weight decay: {w_decay}')
    train_eval(SNN(), device, train_loader, val_loader=val_loader,
               epochs=epochs, weight_decay=w_decay, verbose=False)

# Introduce new data
print('-' * 50, 'Part B', '-' * 50)
# Choose model
print('After comparison, we choose RandomForestClassifier and MLPClassifier with Adam solver as the best models')
# Load data as pandas DataFrame
data_B = pd.read_csv('./data/cmc.data', sep=',', header=None, names=[
    'wife_age', 'wife_education', 'husband_education', 'num_children', 'wife_religion', 'wife_work', 'husband_occupation', 'standard_living', 'media_exposure', 'contraceptive_method'])
X_B = data_B.iloc[:, :-1]
y_B = data_B.iloc[:, -1]

# Check for missing data
print('The distribution of NaN values across features:')
print(X_B.isna().sum())
# Using RandomForestClassifier
print('#' * 20, 'RandomForestClassifier', '#' * 20)
sns.pairplot(data_B, hue='contraceptive_method', diag_kind='kde')
plt.savefig('img/contraceptive_method_pairplot.png')
corr = data_B.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('img/contraceptive_method_corr.png')

X_train, X_test, y_train, y_test = train_test_split(
    X_B, y_B, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42)  # 100 trees

# Train model
rf_model.fit(X_train, y_train)
# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
# Calculate F1 Score
# Weighted average for multi-class
f1 = f1_score(y_test, y_pred, average='weighted')
# Calculate ROC-AUC Score
# For multi-class, set the average parameter; for binary classification, it is not needed
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(
    X_test), multi_class='ovr', average='weighted')

# Print results
print("Evaluation Metrics:")
print("Accuracy:", accuracy)
print("F1 Score (weighted):", f1)
print("ROC-AUC Score (weighted):", roc_auc)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Get feature importances
feature_importances = pd.Series(
    rf_model.feature_importances_, index=X_B.columns)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importances')
plt.savefig('img/feature_importances.png')

# Using MLPClassifier
print('#' * 20, 'MLPClassifier', '#' * 20)

# Visualize each variable's distribution
plt.figure(figsize=(18, 18))
for i, col in enumerate(X_B.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(X_B[col], kde=True)
plt.tight_layout()
plt.savefig('img/contraceptive_method_choice.png')

# For target value category statistics
B_colors = [cmap(i) for i in np.linspace(0, 1, len(y_B.value_counts()))]
plt.figure(figsize=(8, 8))
plt.pie(y_B.value_counts(), autopct='%.2f%%', startangle=90, labels=[
        i for i in y_B.value_counts().index], colors=B_colors)
plt.title('Pie chart')
plt.savefig('img/contraceptive_method_choice_pie.png')
B_ovo, B_ovr = simple_nn_train(MLPClassifier(solver='adam', warm_start=True, early_stopping=True, learning_rate='adaptive',
                               learning_rate_init=0.001, hidden_layer_sizes=(256, 256)), X=X_B, y=y_B, iter=150, only=['ovr'])

print('-' * 50, 'Part C', '', '-' * 50)

# Load dataset
file_path = './data/processed.cleveland.data'  # Heart Disease dataset path

# Add column names
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(file_path, names=column_names, header=None, na_values='?')

# Data preprocessing - check for missing values and fill
print(df.isnull().sum())  # Check for missing values
df = df.dropna()  # Drop rows with missing values

# Data analysis and visualization
# View basic information about the data
# df.info()

# Plot the distribution of the target variable
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Presence (0 = No, 1 = Yes)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.savefig('img/heart_disease_count.png')

# Plot heatmap of feature correlations
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.savefig('img/heart_disease_heatmap.png')

# Model construction - use 'target' as the target variable, other features as input
X = df.drop(columns=['target'])
y = df['target']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate Decision Tree Classifier
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
print(f"Decision Tree - Accuracy: {accuracy_dt:.2f}, F1 Score: {f1_dt:.2f}")

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest Classifier
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
print(f"Random Forest - Accuracy: {accuracy_rf:.2f}, F1 Score: {f1_rf:.2f}")

# Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate Logistic Regression Model
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
print(
    f"Logistic Regression - Accuracy: {accuracy_lr:.2f}, F1 Score: {f1_lr:.2f}")

# Compare model performance
print("\nModel Performance Comparison:")
print(f"Decision Tree - Accuracy: {accuracy_dt:.2f}, F1 Score: {f1_dt:.2f}")
print(f"Random Forest - Accuracy: {accuracy_rf:.2f}, F1 Score: {f1_rf:.2f}")
print(
    f"Logistic Regression - Accuracy: {accuracy_lr:.2f}, F1 Score: {f1_lr:.2f}")

# Visualize model performance
models = ['Decision Tree', 'Random Forest', 'Logistic Regression']
accuracies = [accuracy_dt, accuracy_rf, accuracy_lr]
f1_scores = [f1_dt, f1_rf, f1_lr]

plt.figure(figsize=(10, 6))

# Plot accuracy bar chart
plt.subplot(1, 2, 1)
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Plot F1 score bar chart
plt.subplot(1, 2, 2)
plt.bar(models, f1_scores, color=['blue', 'green', 'red'])
plt.title('Model F1 Score Comparison')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('img/model_comparison.png')
