import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import lightgbm as lgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 1. 数据加载和预处理
def load_data(file_path):
    """加载CSV数据"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """预处理数据"""
    # 确保结果列是字符串类型
    df.iloc[:, 17] = df.iloc[:, 17].astype(str)
    
    # 编码结果列 (Win, Loss, Draw)
    le = LabelEncoder()
    df['result_encoded'] = le.fit_transform(df.iloc[:, 17])
    
    return df, le

# 2. 特征工程 - 为每支队伍创建历史特征
def create_team_features(df):
    """为每支队伍创建历史特征"""
    # 获取所有队伍ID
    team_ids = df.iloc[:, 1].unique()
    
    # 存储每支队伍的历史特征
    team_features = {}
    
    for team_id in team_ids:
        # 获取该队伍的所有比赛记录
        team_matches = df[df.iloc[:, 1] == team_id]
        
        # 计算特征的平均值
        feature_means = team_matches.iloc[:, 2:16].mean().to_dict()
        
        # 添加更多统计特征
        feature_stds = team_matches.iloc[:, 2:16].std().to_dict()
        feature_maxs = team_matches.iloc[:, 2:16].max().to_dict()
        feature_mins = team_matches.iloc[:, 2:16].min().to_dict()
        
        # 组合所有特征
        team_feature = {}
        for key in feature_means:
            team_feature[f'{key}_mean'] = feature_means[key]
            team_feature[f'{key}_std'] = feature_stds[key] if not pd.isna(feature_stds[key]) else 0
            team_feature[f'{key}_max'] = feature_maxs[key]
            team_feature[f'{key}_min'] = feature_mins[key]
        
        # 添加比赛场次
        team_feature['match_count'] = len(team_matches)
        
        # 添加胜率
        win_rate = len(team_matches[team_matches.iloc[:, 17] == 'Win']) / len(team_matches) if len(team_matches) > 0 else 0
        team_feature['win_rate'] = win_rate
        
        # 添加平局率
        draw_rate = len(team_matches[team_matches.iloc[:, 17] == 'Draw']) / len(team_matches) if len(team_matches) > 0 else 0
        team_feature['draw_rate'] = draw_rate
        
        team_features[team_id] = team_feature
    
    return team_features

# 3. 创建比赛级别的特征
def create_match_features(df, team_features):
    """为每场比赛创建特征"""
    match_ids = df.iloc[:, 0].unique()
    match_data = []
    match_labels = []
    
    for match_id in match_ids:
        match = df[df.iloc[:, 0] == match_id]
        
        # 确保每场比赛有6支队伍
        if len(match) != 6:
            continue
            
        # 获取比赛结果 (所有队伍结果相同)
        result = match.iloc[0, 17]
        
        # 分离两个联盟 (假设前3支队伍是一个联盟，后3支是另一个)
        alliance1 = match.iloc[:3, 1].values
        alliance2 = match.iloc[3:, 1].values
        
        # 为每个联盟创建特征
        alliance1_features = create_alliance_features(alliance1, team_features)
        alliance2_features = create_alliance_features(alliance2, team_features)
        
        # 创建差异特征
        diff_features = {}
        for key in alliance1_features:
            if key in alliance2_features:
                diff_features[f'{key}_diff'] = alliance1_features[key] - alliance2_features[key]
        
        # 组合所有特征
        match_feature = {**alliance1_features, **alliance2_features, **diff_features}
        match_data.append(match_feature)
        match_labels.append(result)
    
    return pd.DataFrame(match_data), pd.Series(match_labels)

def create_alliance_features(alliance_teams, team_features):
    """为联盟创建特征"""
    alliance_feature = {}
    
    # 对联盟中每支队伍的特征进行平均
    for i, team_id in enumerate(alliance_teams):
        if team_id not in team_features:
            continue
            
        for feature_name, value in team_features[team_id].items():
            if f'alliance_{feature_name}_mean' not in alliance_feature:
                alliance_feature[f'alliance_{feature_name}_mean'] = 0
                alliance_feature[f'alliance_{feature_name}_max'] = -float('inf')
                alliance_feature[f'alliance_{feature_name}_min'] = float('inf')
                alliance_feature[f'alliance_{feature_name}_sum'] = 0
            
            alliance_feature[f'alliance_{feature_name}_mean'] += value / len(alliance_teams)
            alliance_feature[f'alliance_{feature_name}_max'] = max(alliance_feature[f'alliance_{feature_name}_max'], value)
            alliance_feature[f'alliance_{feature_name}_min'] = min(alliance_feature[f'alliance_{feature_name}_min'], value)
            alliance_feature[f'alliance_{feature_name}_sum'] += value
    
    return alliance_feature

# 4. 模型训练和评估
def train_and_evaluate(X_train, X_test, y_train, y_test, le):
    """训练和评估模型"""
    # 编码标签
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train_encoded)
    test_data = lgb.Dataset(X_test, label=y_test_encoded, reference=train_data)
    
    # 设置参数
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # 训练模型
    model = lgb.train(params,
                     train_data,
                     num_boost_round=1000,
                     valid_sets=[train_data, test_data],
                     callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)])
    
    # 预测
    y_pred_train_proba = model.predict(X_train)
    y_pred_test_proba = model.predict(X_test)
    
    y_pred_train = np.argmax(y_pred_train_proba, axis=1)
    y_pred_test = np.argmax(y_pred_test_proba, axis=1)
    
    # 计算评估指标
    train_accuracy = accuracy_score(y_train_encoded, y_pred_train)
    test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
    
    train_f1 = f1_score(y_train_encoded, y_pred_train, average='weighted')
    test_f1 = f1_score(y_test_encoded, y_pred_test, average='weighted')
    
    train_logloss = log_loss(y_train_encoded, y_pred_train_proba)
    test_logloss = log_loss(y_test_encoded, y_pred_test_proba)
    
    # 计算AUC
    y_train_bin = label_binarize(y_train_encoded, classes=[0, 1, 2])
    y_test_bin = label_binarize(y_test_encoded, classes=[0, 1, 2])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_test_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制AUC曲线
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    class_names = le.classes_
    
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # 打印结果
    print("模型评估结果:")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"训练集F1分数: {train_f1:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")
    print(f"训练集对数损失: {train_logloss:.4f}")
    print(f"测试集对数损失: {test_logloss:.4f}")
    print("\n各类别AUC:")
    for i, class_name in enumerate(le.classes_):
        print(f"{class_name}: {roc_auc[i]:.4f}")
    
    return model, y_pred_test_proba

# 主函数
def main():
    # 加载数据
    print("加载数据...")
    df = load_data('matches.csv')  # 替换为您的CSV文件路径
    
    # 预处理数据
    print("预处理数据...")
    df, le = preprocess_data(df)
    
    # 创建队伍特征
    print("创建队伍历史特征...")
    team_features = create_team_features(df)
    
    # 创建比赛特征
    print("创建比赛级别特征...")
    X, y = create_match_features(df, team_features)
    
    # 处理缺失值
    X = X.fillna(0)
    
    # 划分训练集和测试集
    print("划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练和评估模型
    print("训练和评估模型...")
    model, y_pred_proba = train_and_evaluate(X_train, X_test, y_train, y_test, le)
    
    # 输出预测概率示例
    print("\n测试集前5场比赛的预测概率:")
    for i in range(min(5, len(y_test))):
        print(f"比赛 {i+1}: 实际结果 = {y_test.iloc[i]}")
        for j, class_name in enumerate(le.classes_):
            print(f"  {class_name}概率: {y_pred_proba[i][j]:.4f}")
        print()

if __name__ == "__main__":
    main()