import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 读取数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 数据预处理
def preprocess_data(df):
    # 确保Result列是字符串类型
    df['Result'] = df['Result'].astype(str)
    
    # 提取比赛ID和队伍ID
    match_ids = df['Match'].unique()
    
    # 按比赛分组
    matches = []
    for match_id in match_ids:
        match_data = df[df['Match'] == match_id]
        matches.append(match_data)
    
    return matches

# 计算每支队伍的历史特征
def calculate_team_features(df):
    # 只使用训练数据计算队伍特征
    team_features = defaultdict(list)
    team_stats = {}
    
    # 获取特征列
    feature_columns = df.columns[2:16]  # 第3列到第16列
    
    for _, row in df.iterrows():
        team_id = row['Teams']
        features = row[feature_columns].values.astype(float)
        team_features[team_id].append(features)
    
    # 计算每个队伍的平均特征
    for team_id, features_list in team_features.items():
        avg_features = np.mean(features_list, axis=0)
        team_stats[team_id] = avg_features
    
    return team_stats

# 创建数据集
class MatchDataset(Dataset):
    def __init__(self, matches, team_stats, is_train=True):
        self.matches = matches
        self.team_stats = team_stats
        self.is_train = is_train
        self.features, self.labels = self.process_matches()
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        if len(self.labels) > 0:
            self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        else:
            self.labels_encoded = np.array([])
        
        # 标准化特征
        self.scaler = StandardScaler()
        if len(self.features) > 0:
            self.features_scaled = self.scaler.fit_transform(self.features)
        else:
            self.features_scaled = np.array([])
    
    def process_matches(self):
        features = []
        labels = []
        feature_columns = 14  # 每个队伍有14个特征
        
        for match in self.matches:
            if len(match) != 6:
                continue  # 跳过不完整的比赛
                
            # 获取比赛结果
            results = match['Result'].values
            unique_results = np.unique(results)
            
            if len(unique_results) == 1 and unique_results[0] == 'Draw':
                label = 'Draw'
            else:
                # 找出获胜的队伍
                win_teams = match[match['Result'] == 'Win']['Teams'].values
                if len(win_teams) == 3:
                    label = 'Win'  # 表示第一个联盟获胜
                else:
                    # 如果数据有问题，跳过这场比赛
                    continue
            
            # 提取队伍特征
            match_features = []
            for _, row in match.iterrows():
                team_id = row['Teams']
                if team_id in self.team_stats:
                    match_features.extend(self.team_stats[team_id])
                else:
                    # 如果队伍没有历史数据，使用零向量
                    match_features.extend(np.zeros(feature_columns))
            
            # 检查特征向量长度是否正确
            if len(match_features) == 6 * feature_columns:
                features.append(match_features)
                labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def __len__(self):
        return len(self.features_scaled)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features_scaled[idx])
        label = torch.LongTensor([self.labels_encoded[idx]]) if len(self.labels_encoded) > 0 else torch.LongTensor([0])
        return feature, label

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output_layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        # 但我们需要 (seq_len, batch_size, input_dim) 用于Transformer
        x = x.reshape(x.size(0), -1, self.input_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        
        # 输入投影
        x = self.input_projection(x)  # (seq_len, batch_size, hidden_dim)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)
        
        # 取第一个位置的输出作为分类特征
        x = x[0, :, :]  # (batch_size, hidden_dim)
        
        # 输出层
        x = self.output_layer(x)  # (batch_size, num_classes)
        
        return x

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).squeeze()
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')
    
    return train_losses, val_losses, val_accuracies

# 评估函数
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device).squeeze()
            
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_probs, all_labels

# 绘制AUC曲线
def plot_auc_curve(y_true, y_probs, label_encoder, title='ROC Curve'):
    n_classes = len(label_encoder.classes_)
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(8, 6))
    
    # 为每个类别计算ROC曲线和AUC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制每个类别的ROC曲线
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(label_encoder.inverse_transform([i])[0], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc

# 主函数
def main():
    # 加载数据
    file_path = 'matches.csv'  # 替换为您的文件路径
    df = load_data(file_path)
    
    # 预处理数据
    matches = preprocess_data(df)
    
    # 划分训练集和测试集
    train_matches, test_matches = train_test_split(matches, test_size=0.2, random_state=42)
    
    # 计算队伍特征（仅使用训练数据）
    train_df = pd.concat(train_matches)
    team_stats = calculate_team_features(train_df)
    
    # 创建数据集
    train_dataset = MatchDataset(train_matches, team_stats, is_train=True)
    test_dataset = MatchDataset(test_matches, team_stats, is_train=False)
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    input_dim = 14  # 每个队伍的特征维度
    hidden_dim = 64
    num_classes = len(train_dataset.label_encoder.classes_)
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    
    model = TransformerModel(input_dim, hidden_dim, num_classes, num_heads, num_layers, dropout).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 20
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs
    )
    
    # 评估训练集
    train_preds, train_probs, train_labels = evaluate_model(model, train_loader)
    
    # 评估测试集
    test_preds, test_probs, test_labels = evaluate_model(model, test_loader)
    
    # 计算指标
    # 训练集
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='weighted')
    train_log_loss = log_loss(train_labels, train_probs)
    
    # 测试集
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_log_loss = log_loss(test_labels, test_probs)
    
    print("\n训练集指标:")
    print(f"准确率: {train_accuracy:.4f}")
    print(f"F1分数: {train_f1:.4f}")
    print(f"对数损失: {train_log_loss:.4f}")
    
    print("\n测试集指标:")
    print(f"准确率: {test_accuracy:.4f}")
    print(f"F1分数: {test_f1:.4f}")
    print(f"对数损失: {test_log_loss:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # 绘制测试集的AUC曲线
    test_probs_array = np.array(test_probs)
    roc_auc = plot_auc_curve(np.array(test_labels), test_probs_array, 
                            train_dataset.label_encoder, title='Test Set ROC Curve')
    
    # 打印每个类别的AUC
    for i, class_name in enumerate(train_dataset.label_encoder.classes_):
        print(f"类别 {class_name} 的AUC: {roc_auc[i]:.4f}")

if __name__ == "__main__":
    main()