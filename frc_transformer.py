import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_csv('matches.csv')
features_columns = data.columns[2:16]  # C到P列是特征

# 计算每支队伍的历史平均特征
def compute_historical_averages(data, teams):
    historical_avgs = {}
    for team in teams:
        team_data = data[data['Teams'] == team]
        if len(team_data) > 0:
            historical_avgs[team] = team_data[features_columns].mean().values
        else:
            historical_avgs[team] = np.zeros(len(features_columns))
    return historical_avgs

# 构建比赛数据 - 修改为按比赛ID分组而不是随机划分
def build_match_data_with_teams(data, historical_avgs):
    matches = []
    results = []
    match_teams_list = []  # 存储每场比赛的队伍编号
    match_ids = data['Match'].unique()
    
    for match_id in match_ids:
        match_data = data[data['Match'] == match_id]
        if len(match_data) != 6:
            continue
        team_ids = match_data['Teams'].values
        result = match_data['Result'].values[0]  # 同一场比赛结果相同
        feature_vector = []
        for team in team_ids:
            feature_vector.extend(historical_avgs[team])
        matches.append(feature_vector)
        results.append(result)
        match_teams_list.append(team_ids)  # 保存队伍编号
    
    return np.array(matches), np.array(results), match_teams_list

# 编码标签
le = LabelEncoder()
all_results = data['Result'].unique()
le.fit(all_results)
n_classes = len(le.classes_)
class_names = le.classes_

# 按比赛ID划分训练集和测试集，而不是随机划分行
match_ids = data['Match'].unique()
train_match_ids, test_match_ids = train_test_split(match_ids, test_size=0.2, random_state=42)

# 创建训练集和测试集
train_data = data[data['Match'].isin(train_match_ids)]
test_data = data[data['Match'].isin(test_match_ids)]

print(f"训练集比赛场数: {len(train_match_ids)}")
print(f"测试集比赛场数: {len(test_match_ids)}")

train_teams = train_data['Teams'].unique()
test_teams = test_data['Teams'].unique()

# 计算历史平均特征
historical_avgs_train = compute_historical_averages(train_data, train_teams)
historical_avgs_test = compute_historical_averages(train_data, test_teams)  # 使用训练集的历史平均

# 构建训练和测试特征及标签
X_train, y_train, train_teams_list = build_match_data_with_teams(train_data, historical_avgs_train)
X_test, y_test, test_teams_list = build_match_data_with_teams(test_data, historical_avgs_test)

# 编码标签
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled).view(-1, 6, 14)
X_test_tensor = torch.FloatTensor(X_test_scaled).view(-1, 6, 14)
y_train_tensor = torch.LongTensor(y_train_encoded)
y_test_tensor = torch.LongTensor(y_test_encoded)

# 定义数据集
class MatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MatchDataset(X_train_tensor, y_train_tensor)
test_dataset = MatchDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 6, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear_out = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.linear_in(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(2)
        x = self.linear_out(x)
        return x

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(input_dim=14, d_model=64, nhead=8, dim_feedforward=256, num_layers=2, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# 评估函数 - 修改为返回每场比赛的预测结果
def evaluate_model_with_details(model, data_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_indices = []  # 保存批次和索引信息
    
    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            # 保存批次和索引信息
            for i in range(len(batch_y)):
                all_indices.append((batch_idx, i))
    
    return all_preds, all_probs, all_labels, all_indices

# 计算评估指标
def compute_metrics(y_true, y_pred, y_probs, n_classes):
    # 确保y_true是numpy数组
    y_true = np.array(y_true)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # 处理log_loss计算，当只有一个类别时返回NaN
    try:
        loss = log_loss(y_true, y_probs)
    except ValueError:
        loss = float('nan')
    
    # AUC计算（OvR）
    auc_scores = []
    fpr_dict = {}
    tpr_dict = {}
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        # 检查是否至少有一个正样本和一个负样本
        if len(np.unique(y_true_binary)) < 2:
            auc_scores.append(0.5)
            fpr_dict[i] = [0, 1]
            tpr_dict[i] = [0, 1]
            continue
            
        y_prob_binary = [p[i] for p in y_probs]
        try:
            auc_score = roc_auc_score(y_true_binary, y_prob_binary)
            auc_scores.append(auc_score)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
        except ValueError:
            # 处理只有一个类别的情况
            auc_scores.append(0.5)
            fpr_dict[i] = [0, 1]
            tpr_dict[i] = [0, 1]
    
    # 宏观AUC和微观AUC
    y_true_onehot = np.eye(n_classes)[y_true]
    try:
        macro_auc = roc_auc_score(y_true_onehot, y_probs, average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true_onehot, y_probs, average='micro', multi_class='ovr')
    except ValueError:
        macro_auc = 0.5
        micro_auc = 0.5
    
    return accuracy, f1, loss, auc_scores, macro_auc, micro_auc, fpr_dict, tpr_dict

# 在训练集和测试集上评估
train_preds, train_probs, train_labels, train_indices = evaluate_model_with_details(model, train_loader, device)
test_preds, test_probs, test_labels, test_indices = evaluate_model_with_details(model, test_loader, device)

train_accuracy, train_f1, train_loss, train_aucs, train_macro_auc, train_micro_auc, train_fpr, train_tpr = compute_metrics(train_labels, train_preds, train_probs, n_classes)
test_accuracy, test_f1, test_loss, test_aucs, test_macro_auc, test_micro_auc, test_fpr, test_tpr = compute_metrics(test_labels, test_preds, test_probs, n_classes)

print("Training Metrics:")
print(f"Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Log Loss: {train_loss:.4f}")
print(f"Class AUCs: {[f'{auc:.4f}' for auc in train_aucs]}")
print(f"Macro AUC: {train_macro_auc:.4f}, Micro AUC: {train_micro_auc:.4f}")

print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Log Loss: {test_loss:.4f}")
print(f"Class AUCs: {[f'{auc:.4f}' for auc in test_aucs]}")
print(f"Macro AUC: {test_macro_auc:.4f}, Micro AUC: {test_micro_auc:.4f}")

# 输出前20个验证结果，包含队伍编号
print("\n前20个训练集验证结果:")
print("预测结果\t实际结果\t队伍编号")
for i in range(min(20, len(train_preds))):
    pred_label = class_names[train_preds[i]]
    true_label = class_names[train_labels[i]]
    teams = train_teams_list[i]  # 获取队伍编号
    print(f"{pred_label}\t\t{true_label}\t\t{teams}")

print("\n前20个测试集验证结果:")
print("预测结果\t实际结果\t队伍编号")
for i in range(min(20, len(test_preds))):
    pred_label = class_names[test_preds[i]]
    true_label = class_names[test_labels[i]]
    teams = test_teams_list[i]  # 获取队伍编号
    print(f"{pred_label}\t\t{true_label}\t\t{teams}")

# 绘制AUC曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(n_classes):
    plt.plot(train_fpr[i], train_tpr[i], label=f'Class {class_names[i]} (AUC = {train_aucs[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Training Set ROC Curves')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(n_classes):
    plt.plot(test_fpr[i], test_tpr[i], label=f'Class {class_names[i]} (AUC = {test_aucs[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Set ROC Curves')
plt.legend()
plt.tight_layout()
plt.show()