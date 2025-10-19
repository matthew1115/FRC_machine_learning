import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, 
    roc_curve, auc, roc_auc_score, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
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
        if len(match_data) == 6:  # 只保留完整的比赛
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
    def __init__(self, matches, team_stats, label_encoder=None, scaler=None):
        self.matches = matches
        self.team_stats = team_stats
        self.features, self.labels = self.process_matches()
        
        # 编码标签
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        else:
            self.label_encoder = label_encoder
            # 处理测试集中可能出现的未知标签
            self.labels_encoded = []
            for label in self.labels:
                if label in self.label_encoder.classes_:
                    self.labels_encoded.append(self.label_encoder.transform([label])[0])
                else:
                    # 将未知标签映射为最常见的标签
                    self.labels_encoded.append(0)  # 假设0是最常见的标签
        
        # 标准化特征
        if scaler is None:
            self.scaler = StandardScaler()
            self.features_scaled = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features_scaled = self.scaler.transform(self.features)
    
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
            
            # 处理三种可能的结果: Win, Loss, Draw
            if len(unique_results) == 1:
                if unique_results[0] == 'Draw':
                    label = 'Draw'
                elif unique_results[0] == 'Win':
                    # 这种情况不应该发生，因为每场比赛应该有Win和Loss
                    print(f"警告: 比赛 {match['Match'].iloc[0]} 只有Win结果")
                    continue
                elif unique_results[0] == 'Loss':
                    # 这种情况不应该发生，因为每场比赛应该有Win和Loss
                    print(f"警告: 比赛 {match['Match'].iloc[0]} 只有Loss结果")
                    continue
                else:
                    # 未知结果
                    print(f"警告: 比赛 {match['Match'].iloc[0]} 有未知结果: {unique_results[0]}")
                    continue
            elif len(unique_results) == 2:
                if 'Win' in unique_results and 'Loss' in unique_results:
                    # 正常比赛，确定哪个联盟获胜
                    win_teams = match[match['Result'] == 'Win']['Teams'].values
                    loss_teams = match[match['Result'] == 'Loss']['Teams'].values
                    
                    if len(win_teams) == 3 and len(loss_teams) == 3:
                        # 正常情况，3支队伍Win，3支队伍Loss
                        # 我们选择第一个Win队伍所在的联盟作为参考
                        # 注意: 这里我们假设前3支队伍是一个联盟
                        first_team_result = match['Result'].iloc[0]
                        label = first_team_result
                    else:
                        print(f"警告: 比赛 {match['Match'].iloc[0]} Win/Loss队伍数量不匹配")
                        continue
                else:
                    print(f"警告: 比赛 {match['Match'].iloc[0]} 有异常结果组合: {unique_results}")
                    continue
            else:
                print(f"警告: 比赛 {match['Match'].iloc[0]} 有多于2种结果: {unique_results}")
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
        label = torch.LongTensor([self.labels_encoded[idx]])
        return feature, label

# 特征标记化层
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(num_features, d_token))
        nn.init.normal_(self.weight, mean=0, std=0.02)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # x shape: [batch_size, num_features]
        # 输出: [batch_size, num_features, d_token]
        return x.unsqueeze(2) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

# FT-Transformer模型
class FTTransformer(nn.Module):
    def __init__(self, num_features, d_token, n_layers, n_heads, d_ff, 
                 dropout=0.1, activation="relu", n_classes=3):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        
        # 特征标记化
        self.feature_tokenizer = FeatureTokenizer(num_features, d_token)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # 位置编码
        self.position_emb = nn.Parameter(torch.randn(1, num_features + 1, d_token))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_token, n_classes)
        
    def forward(self, x):
        # x shape: [batch_size, num_features]
        batch_size = x.shape[0]
        
        # 特征标记化
        x = self.feature_tokenizer(x)  # [batch_size, num_features, d_token]
        
        # 添加[CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_features+1, d_token]
        
        # 添加位置编码
        x = x + self.position_emb
        
        # Transformer编码器
        x = self.transformer_encoder(x)  # [batch_size, num_features+1, d_token]
        
        # 使用[CLS] token的输出进行分类
        cls_output = x[:, 0, :]  # [batch_size, d_token]
        
        # 输出层
        output = self.output_layer(cls_output)  # [batch_size, n_classes]
        
        return output

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

# 计算所有指标
def calculate_metrics(y_true, y_pred, y_probs, label_encoder, dataset_name=""):
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    loss = log_loss(y_true, y_probs)
    
    print(f"\n{dataset_name}指标:")
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"对数损失: {loss:.4f}")
    
    # 分类报告
    print(f"\n{dataset_name}分类报告:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'{dataset_name}混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    # 计算AUC (One-vs-Rest)
    n_classes = len(label_encoder.classes_)
    
    # 检查概率矩阵的形状
    y_probs = np.array(y_probs)
    if y_probs.shape[1] != n_classes:
        print(f"警告: 概率矩阵形状 {y_probs.shape} 与类别数 {n_classes} 不匹配")
        print("无法计算AUC，跳过ROC曲线绘制")
        return {
            'accuracy': accuracy,
            'f1': f1,
            'log_loss': loss,
            'roc_auc': None
        }
    
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # 计算每个类别的AUC
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception as e:
            print(f"计算类别 {i} 的ROC曲线时出错: {e}")
            roc_auc[i] = 0.5  # 中性AUC值
    
    # 计算宏观平均AUC
    try:
        fpr["macro"], tpr["macro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    except Exception as e:
        print(f"计算宏观平均AUC时出错: {e}")
        roc_auc["macro"] = 0.5
    
    # 计算微观平均AUC
    try:
        roc_auc["micro"] = roc_auc_score(y_true_bin, y_probs, average="micro", multi_class="ovr")
    except Exception as e:
        print(f"计算微观平均AUC时出错: {e}")
        roc_auc["micro"] = 0.5
    
    print(f"\n{dataset_name}AUC值:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"类别 {class_name} 的AUC: {roc_auc[i]:.4f}")
    print(f"宏观平均AUC: {roc_auc['macro']:.4f}")
    print(f"微观平均AUC: {roc_auc['micro']:.4f}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, color in zip(range(n_classes), colors):
        if i in roc_auc and roc_auc[i] != 0.5:  # 只绘制成功计算的曲线
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.4f})'
                     ''.format(label_encoder.classes_[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} ROC曲线 (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'log_loss': loss,
        'roc_auc': roc_auc
    }

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, val_accuracies):
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
    
    # 创建训练数据集
    train_dataset = MatchDataset(train_matches, team_stats)
    
    # 检查训练集中的类别分布
    unique, counts = np.unique(train_dataset.labels_encoded, return_counts=True)
    print("训练集类别分布:")
    for cls, count in zip(unique, counts):
        print(f"类别 {train_dataset.label_encoder.inverse_transform([cls])[0]}: {count} 个样本")
    
    # 创建测试数据集（使用训练集的标签编码器和标准化器）
    test_dataset = MatchDataset(
        test_matches, team_stats, 
        label_encoder=train_dataset.label_encoder,
        scaler=train_dataset.scaler
    )
    
    # 检查测试集中的类别分布
    unique, counts = np.unique(test_dataset.labels_encoded, return_counts=True)
    print("测试集类别分布:")
    for cls, count in zip(unique, counts):
        print(f"类别 {test_dataset.label_encoder.inverse_transform([cls])[0]}: {count} 个样本")
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化FT-Transformer模型
    num_features = 6 * 14  # 6支队伍 * 14个特征
    d_token = 32  # 标记维度
    n_layers = 3  # Transformer层数
    n_heads = 4   # 注意力头数
    d_ff = 256    # 前馈网络维度
    dropout = 0.1
    n_classes = len(train_dataset.label_encoder.classes_)
    
    print(f"FT-Transformer配置: 特征数={num_features}, 标记维度={d_token}, 类别数={n_classes}")
    print(f"层数={n_layers}, 头数={n_heads}, 前馈维度={d_ff}")
    
    model = FTTransformer(
        num_features=num_features,
        d_token=d_token,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        n_classes=n_classes
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练模型
    num_epochs = 20
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    # 评估训练集
    train_preds, train_probs, train_labels = evaluate_model(model, train_loader)
    
    # 评估测试集
    test_preds, test_probs, test_labels = evaluate_model(model, test_loader)
    
    # 检查概率矩阵形状
    print(f"训练集概率矩阵形状: {np.array(train_probs).shape}")
    print(f"测试集概率矩阵形状: {np.array(test_probs).shape}")
    
    # 计算指标
    train_metrics = calculate_metrics(
        train_labels, train_preds, np.array(train_probs), 
        train_dataset.label_encoder, "训练集"
    )
    
    test_metrics = calculate_metrics(
        test_labels, test_preds, np.array(test_probs), 
        test_dataset.label_encoder, "测试集"
    )
    
    # 打印模型总结
    print("\n模型总结:")
    print(f"训练集准确率: {train_metrics['accuracy']:.4f}")
    print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    print(f"训练集F1分数: {train_metrics['f1']:.4f}")
    print(f"测试集F1分数: {test_metrics['f1']:.4f}")
    print(f"训练集对数损失: {train_metrics['log_loss']:.4f}")
    print(f"测试集对数损失: {test_metrics['log_loss']:.4f}")
    
    if test_metrics['roc_auc'] is not None:
        print(f"测试集宏观平均AUC: {test_metrics['roc_auc']['macro']:.4f}")
        print(f"测试集微观平均AUC: {test_metrics['roc_auc']['micro']:.4f}")

if __name__ == "__main__":
    main()