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

# 读取数据
data = pd.read_csv('matches.csv')
print("数据读取完成")

# 按每6行分组处理比赛
matches = []
n_matches = len(data) // 6
for i in range(n_matches):
    match_data = data.iloc[i*6:(i+1)*6]
    matches.append(match_data)

# 划分训练集和测试集
train_matches, test_matches = train_test_split(matches, test_size=0.2, random_state=42)
print(f"训练集比赛数: {len(train_matches)}, 测试集比赛数: {len(test_matches)}")

# 提取所有队伍编号
all_teams = pd.concat([match['Teams'] for match in matches]).unique()

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
        features.extend(team_features[team])
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

# 创建验证集 (从训练集中划分)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train_final, y_train_final)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义更简单的MLP模型，添加更多正则化
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32], dropout_rate=0.5):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 添加批归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # 增加dropout比例
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 检查输入维度
input_dim = X_train_tensor.shape[1]
print(f"输入维度: {input_dim}")

# 创建MLP模型
model = MLP(input_dim=input_dim, num_classes=3)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

criterion = nn.CrossEntropyLoss()
# 添加权重衰减(L2正则化)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# 使用学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 训练模型，添加早停机制
num_epochs = 100
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # 学习率调度
    scheduler.step(avg_val_loss)
    
    # 早停机制
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()

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
y_val_pred, y_val_true = predict(model, val_loader)
y_test_pred, y_test_true = predict(model, test_loader)

# 计算评估指标
def evaluate(y_true, y_pred):
    y_pred_label = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label, average='weighted')
    loss = log_loss(y_true, y_pred)
    
    # AUC计算
    auc_scores = []
    for i in range(3):
        auc_score = roc_auc_score((y_true == i).astype(int), y_pred[:, i])
        auc_scores.append(auc_score)
    macro_auc = np.mean(auc_scores)
    micro_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')
    
    return accuracy, f1, loss, auc_scores, macro_auc, micro_auc

train_accuracy, train_f1, train_loss, train_auc, train_macro_auc, train_micro_auc = evaluate(y_train_true, y_train_pred)
val_accuracy, val_f1, val_loss, val_auc, val_macro_auc, val_micro_auc = evaluate(y_val_true, y_val_pred)
test_accuracy, test_f1, test_loss, test_auc, test_macro_auc, test_micro_auc = evaluate(y_test_true, y_test_pred)

print("训练集评估指标:")
print(f"准确率: {train_accuracy:.4f}, F1分数: {train_f1:.4f}, 对数损失: {train_loss:.4f}")
print(f"各类别AUC: {[f'{auc:.4f}' for auc in train_auc]}")
print(f"宏观AUC: {train_macro_auc:.4f}, 微观AUC: {train_micro_auc:.4f}")

print("\n验证集评估指标:")
print(f"准确率: {val_accuracy:.4f}, F1分数: {val_f1:.4f}, 对数损失: {val_loss:.4f}")
print(f"各类别AUC: {[f'{auc:.4f}' for auc in val_auc]}")
print(f"宏观AUC: {val_macro_auc:.4f}, 微观AUC: {val_micro_auc:.4f}")

print("\n测试集评估指标:")
print(f"准确率: {test_accuracy:.4f}, F1分数: {test_f1:.4f}, 对数损失: {test_loss:.4f}")
print(f"各类别AUC: {[f'{auc:.4f}' for auc in test_auc]}")
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
    #          label=f'Macro-average (AUC = {macro_auc:.2f})')
    
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
plot_auc_curves(y_val_true, y_val_pred, 'Validation ROC Curves')
plot_auc_curves(y_test_true, y_test_pred, 'Test ROC Curves')

# 输出前20个预测结果和实际结果
print("\n训练集前20个预测结果:")
for i in range(min(20, len(y_train_true))):
    pred_label = np.argmax(y_train_pred[i])
    true_label = y_train_true[i]
    print(f"样本 {i+1}: 预测={pred_label}({'Win' if pred_label==0 else 'Loss' if pred_label==1 else 'Draw'}), "
          f"实际={true_label}({'Win' if true_label==0 else 'Loss' if true_label==1 else 'Draw'})")

print("\n验证集前20个预测结果:")
for i in range(min(20, len(y_val_true))):
    pred_label = np.argmax(y_val_pred[i])
    true_label = y_val_true[i]
    print(f"样本 {i+1}: 预测={pred_label}({'Win' if pred_label==0 else 'Loss' if pred_label==1 else 'Draw'}), "
          f"实际={true_label}({'Win' if true_label==0 else 'Loss' if true_label==1 else 'Draw'})")

print("\n测试集前20个预测结果:")
for i in range(min(20, len(y_test_true))):
    pred_label = np.argmax(y_test_pred[i])
    true_label = y_test_true[i]
    print(f"样本 {i+1}: 预测={pred_label}({'Win' if pred_label==0 else 'Loss' if pred_label==1 else 'Draw'}), "
          f"实际={true_label}({'Win' if true_label==0 else 'Loss' if true_label==1 else 'Draw'})")

# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, title):
    y_pred_label = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Win', 'Loss', 'Draw'])
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_train_true, y_train_pred, 'Train Confusion Matrix')
plot_confusion_matrix(y_val_true, y_val_pred, 'Validation Confusion Matrix')
plot_confusion_matrix(y_test_true, y_test_pred, 'Test Confusion Matrix')