import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 读取测试数据
data = pd.read_csv('playoffs.csv')
print("playoffs数据读取完成")

# 按每6行分组处理比赛
test_matches = []
n_matches = len(data) // 6
for i in range(n_matches):
    match_data = data.iloc[i*6:(i+1)*6]
    test_matches.append(match_data)

# 读取训练数据
data = pd.read_csv('matches.csv')
print("matches数据读取完成")

# 按每6行分组处理比赛
train_matches = []
n_matches = len(data) // 6
for i in range(n_matches):
    match_data = data.iloc[i*6:(i+1)*6]
    train_matches.append(match_data)

print(f"训练集比赛数: {len(train_matches)}, 测试集比赛数: {len(test_matches)}")

# 提取所有队伍编号
all_teams = pd.concat([match['Teams'] for match in train_matches]).unique()

# 优化后的历史平均特征计算
print("开始计算队伍历史平均特征...")

# 首先，创建一个字典来存储每个队伍的所有特征数据
team_feature_dict = {team: [] for team in all_teams}

# 一次性遍历所有训练比赛，收集每个队伍的特征数据
for match in train_matches:
    for _, row in match.iterrows():
        team = row['Teams']
        features = row.iloc[2:16].values  # 直接获取14个特征值
        if len(features) == 14:
            team_feature_dict[team].append(features)

# 然后计算每个队伍的平均特征
team_features = {}
for team, features_list in team_feature_dict.items():
    if features_list:
        # 将所有特征堆叠成一个数组
        features_array = np.vstack(features_list)
        team_features[team] = np.mean(features_array, axis=0)
    else:
        team_features[team] = np.zeros(14)

print("队伍历史平均特征计算完成")

# 构建训练集特征和标签
X_train, y_train = [], []
for match in train_matches:
    teams = match['Teams'].values
    results = match['Result'].values
    features = []
    for team in teams:
        features.extend(team_features[team])
    X_train.append(features)
    
    # 确定比赛结果
    if all(result == 'Win' for result in results[:3]) and all(result == 'Loss' for result in results[3:]):
        y_train.append(0)  # 前三个队伍获胜
    elif all(result == 'Loss' for result in results[:3]) and all(result == 'Win' for result in results[3:]):
        y_train.append(1)  # 后三个队伍获胜
    else:
        y_train.append(2)  # 平局

X_train = np.array(X_train)
y_train = np.array(y_train)

# 构建测试集特征和标签
X_test, y_test = [], []
for match in test_matches:
    teams = match['Teams'].values
    results = match['Result'].values
    features = []
    for team in teams:
        try:
            features.extend(team_features[team])
        except:
            print(f"训练集无此Teams: {teams}")
            print(f"测试集: {match}")
            exit()
			
    X_test.append(features)
    
    # 确定比赛结果
    if all(result == 'Win' for result in results[:3]) and all(result == 'Loss' for result in results[3:]):
        y_test.append(0)  # 前三个队伍获胜
    elif all(result == 'Loss' for result in results[:3]) and all(result == 'Win' for result in results[3:]):
        y_test.append(1)  # 后三个队伍获胜
    else:
        y_test.append(2)  # 平局

X_test = np.array(X_test)
y_test = np.array(y_test)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义FT-Transformer模型
class FTTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=4, num_heads=4, dropout=0.01):
        super(FTTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 添加一个额外的线性层来处理输入
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 创建位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x形状: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # 投影到隐藏维度
        x = self.input_projection(x)  # (batch_size, hidden_dim)
        
        # 添加序列维度并重复位置编码
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        x = x + self.pos_encoding
        
        # 通过Transformer
        x = self.transformer(x)  # (batch_size, 1, hidden_dim)
        
        # 移除序列维度
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        
        # 应用dropout
        x = self.dropout(x)
        
        # 输出层
        x = self.fc(x)  # (batch_size, num_classes)
        return x

# 检查输入维度
print(f"输入维度: {X_train_tensor.shape[1]}")
model = FTTransformer(input_dim=X_train_tensor.shape[1], num_classes=3)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

# 预测函数
def predict(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            predictions.extend(torch.softmax(outputs, dim=1).numpy())
            true_labels.extend(batch_y.numpy())
    return np.array(predictions), np.array(true_labels)

y_train_pred, y_train_true = predict(model, train_loader)
y_test_pred, y_test_true = predict(model, test_loader)

# 计算评估指标
def evaluate(y_true, y_pred):
    y_pred_label = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label, average='weighted')

	#判断是否出现平局
    #loss = 0
    #loss = log_loss(y_true, y_pred, labels=[0, 1, 2])
    
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 3:  
        print("无平局出现")
        loss = log_loss(y_true, y_pred, labels=[0, 1, 2])
    else:
        print("存在平局出现")
        loss = log_loss(y_true, y_pred)
    
    # AUC计算
    auc_scores = []
    for i in range(3):
        auc_score = roc_auc_score((y_true == i).astype(int), y_pred[:, i])
        auc_scores.append(auc_score)
    macro_auc = np.mean(auc_scores)
    #micro_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')
    if len(unique_classes) < 3:  
        micro_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro', labels=[0, 1, 2])
    else:
        micro_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')
    return accuracy, f1, loss, auc_scores, macro_auc, micro_auc

train_accuracy, train_f1, train_loss, train_auc, train_macro_auc, train_micro_auc = evaluate(y_train_true, y_train_pred)
test_accuracy, test_f1, test_loss, test_auc, test_macro_auc, test_micro_auc = evaluate(y_test_true, y_test_pred)

print("训练集评估指标:")
print(f"准确率: {train_accuracy:.4f}, F1分数: {train_f1:.4f}, 对数损失: {train_loss:.4f}")
print(f"各类别AUC: {train_auc}")
print(f"宏观AUC: {train_macro_auc:.4f}, 微观AUC: {train_micro_auc:.4f}")

print("测试集评估指标:")
print(f"准确率: {test_accuracy:.4f}, F1分数: {test_f1:.4f}, 对数损失: {test_loss:.4f}")
print(f"各类别AUC: {test_auc}")
print(f"宏观AUC: {test_macro_auc:.4f}, 微观AUC: {test_micro_auc:.4f}")

# 绘制AUC曲线
def plot_auc_curves(y_true, y_pred, title):
    plt.figure(figsize=(10, 8))
    colors = ['green', 'orange', 'blue']
    labels = ['Class Win', 'Class Loss', 'Class Draw']
    
    for i in range(3):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                 label=f'{labels[i]} (AUC = {roc_auc:.2f})')
    
    # 计算宏观AUC
    all_fpr = np.unique(np.concatenate([fpr for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= 3
    macro_auc = auc(all_fpr, mean_tpr)
    # plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', lw=2, 
    #         label=f'Macro-average (AUC = {macro_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

plot_auc_curves(y_train_true, y_train_pred, 'Train ROC Curves')
plot_auc_curves(y_test_true, y_test_pred, 'Test ROC Curves')

"""
# 输出前20个预测结果和实际结果
print("\n训练集前20个预测结果:")
for i in range(min(20, len(y_train_true))):
    pred_label = np.argmax(y_train_pred[i])
    true_label = y_train_true[i]
    print(f"样本 {i+1}: 预测={pred_label}({'Win' if pred_label==0 else 'Loss' if pred_label==1 else 'Draw'}), "
          f"实际={true_label}({'Win' if true_label==0 else 'Loss' if true_label==1 else 'Draw'})")
"""
print("\n测试集前20个预测结果:")
for i in range(min(20, len(y_test_true))):
    pred_label = np.argmax(y_test_pred[i])
    true_label = y_test_true[i]
    print(f"样本 {i+1}: 预测={pred_label}({'Win' if pred_label==0 else 'Loss' if pred_label==1 else 'Draw'}), "
          f"实际={true_label}({'Win' if true_label==0 else 'Loss' if true_label==1 else 'Draw'})")
