import numpy as np
import pandas as pd
import lightgbm as lgb
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
import joblib
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
np.random.seed(42)

# 读取数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 数据预处理
def preprocess_data(df):
    df['Result'] = df['Result'].astype(str)
    match_ids = df['Match'].unique()
    matches = []
    for match_id in match_ids:
        match_data = df[df['Match'] == match_id]
        if len(match_data) == 6:
            matches.append(match_data)
    return matches

# 计算每支队伍的历史特征统计量
def calculate_team_features(df, feature_columns):
    team_features = defaultdict(list)
    team_stats = {}
    
    for _, row in df.iterrows():
        team_id = row['Teams']
        features = row[feature_columns].values.astype(float)
        team_features[team_id].append(features)
    
    # 计算每个队伍的特征统计量
    for team_id, features_list in team_features.items():
        if len(features_list) > 0:
            avg_features = np.mean(features_list, axis=0)
            std_features = np.std(features_list, axis=0)
            max_features = np.max(features_list, axis=0)
            min_features = np.min(features_list, axis=0)
            
            team_stats[team_id] = {
                'mean': avg_features,
                'std': std_features,
                'max': max_features,
                'min': min_features,
                'count': len(features_list)
            }
        else:
            # 如果没有数据，使用零向量
            team_stats[team_id] = {
                'mean': np.zeros(len(feature_columns)),
                'std': np.zeros(len(feature_columns)),
                'max': np.zeros(len(feature_columns)),
                'min': np.zeros(len(feature_columns)),
                'count': 0
            }
    
    return team_stats

# 准备训练数据 - 使用历史统计特征
def prepare_data(matches, team_stats, feature_columns):
    features = []
    labels = []
    
    for match in matches:
        if len(match) != 6:
            continue
            
        # 获取比赛结果
        results = match['Result'].values
        unique_results = np.unique(results)
        
        if len(unique_results) == 1:
            if unique_results[0] == 'Draw':
                label = 'Draw'
            else:
                continue
        elif len(unique_results) == 2:
            if 'Win' in unique_results and 'Loss' in unique_results:
                win_teams = match[match['Result'] == 'Win']['Teams'].values
                loss_teams = match[match['Result'] == 'Loss']['Teams'].values
                
                if len(win_teams) == 3 and len(loss_teams) == 3:
                    first_team_result = match['Result'].iloc[0]
                    label = first_team_result
                else:
                    continue
            else:
                continue
        else:
            continue
        
        # 提取特征 - 使用历史统计特征
        match_features = []
        for _, row in match.iterrows():
            team_id = row['Teams']
            if team_id in team_stats:
                stats = team_stats[team_id]
                # 添加多种统计量
                match_features.extend(stats['mean'])  # 平均值
                match_features.extend(stats['std'])   # 标准差
                match_features.extend(stats['max'])   # 最大值
                match_features.extend(stats['min'])   # 最小值
                match_features.append(stats['count'] / 100)  # 样本数量（归一化）
            else:
                # 如果队伍没有历史数据，使用零向量
                match_features.extend(np.zeros(len(feature_columns) * 4 + 1))
        
        features.append(match_features)
        labels.append(label)
    
    return np.array(features), np.array(labels)

# 计算所有指标
def calculate_metrics(y_true, y_pred, y_probs, label_encoder, dataset_name=""):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    loss = log_loss(y_true, y_probs)
    
    print(f"\n{dataset_name}指标:")
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"对数损失: {loss:.4f}")
    
    print(f"\n{dataset_name}分类报告:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'{dataset_name}混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    n_classes = len(label_encoder.classes_)
    y_probs = np.array(y_probs)
    
    if y_probs.shape[1] != n_classes:
        print(f"警告: 概率矩阵形状 {y_probs.shape} 与类别数 {n_classes} 不匹配")
        return {
            'accuracy': accuracy,
            'f1': f1,
            'log_loss': loss,
            'roc_auc': None
        }
    
    roc_auc = {}
    for i in range(n_classes):
        try:
            roc_auc[i] = roc_auc_score((y_true == i).astype(int), y_probs[:, i])
        except Exception as e:
            print(f"计算类别 {i} 的AUC时出错: {e}")
            roc_auc[i] = 0.5
    
    try:
        roc_auc["macro"] = roc_auc_score(y_true, y_probs, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"计算宏观平均AUC时出错: {e}")
        roc_auc["macro"] = 0.5
    
    try:
        roc_auc["micro"] = roc_auc_score(y_true, y_probs, average="micro", multi_class="ovr")
    except Exception as e:
        print(f"计算微观平均AUC时出错: {e}")
        roc_auc["micro"] = 0.5
    
    print(f"\n{dataset_name}AUC值:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"类别 {class_name} 的AUC: {roc_auc[i]:.4f}")
    print(f"宏观平均AUC: {roc_auc['macro']:.4f}")
    print(f"微观平均AUC: {roc_auc['micro']:.4f}")
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        plt.plot(fpr, tpr, color=colors[i], lw=2,
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

# 特征重要性可视化
def plot_feature_importance(model, feature_names):
    importance = model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_imp.head(20))
    plt.title('LightGBM Feature Importance (Top 20)')
    plt.tight_layout()
    plt.show()
    
    return feature_imp

# 主函数
def main():
    # 加载数据
    file_path = 'matches2.csv'
    df = load_data(file_path)
    
    # 定义特征列
    feature_columns = df.columns[2:16]  # 第3列到第16列
    
    # 预处理数据
    matches = preprocess_data(df)
    
    # 划分训练集和测试集
    train_matches, test_matches = train_test_split(matches, test_size=0.2, random_state=42)
    
    # 计算队伍特征（仅使用训练数据）
    train_df = pd.concat(train_matches)
    team_stats = calculate_team_features(train_df, feature_columns)
    
    # 准备训练数据 - 使用历史统计特征
    X_train, y_train = prepare_data(train_matches, team_stats, feature_columns)
    
    # 准备测试数据 - 使用历史统计特征
    X_test, y_test = prepare_data(test_matches, team_stats, feature_columns)
    
    # 检查数据形状
    print(f"训练数据形状: {X_train.shape}")
    print(f"测试数据形状: {X_test.shape}")
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # 检查类别分布
    unique, counts = np.unique(y_train_encoded, return_counts=True)
    print("训练集类别分布:")
    for cls, count in zip(unique, counts):
        print(f"类别 {label_encoder.inverse_transform([cls])[0]}: {count} 个样本")
    
    unique, counts = np.unique(y_test_encoded, return_counts=True)
    print("测试集类别分布:")
    for cls, count in zip(unique, counts):
        print(f"类别 {label_encoder.inverse_transform([cls])[0]}: {count} 个样本")
    
    # 创建特征名称
    feature_names = []
    stat_types = ['Mean', 'Std', 'Max', 'Min']
    for i in range(6):  # 6支队伍
        for stat in stat_types:
            for j in range(len(feature_columns)):  # 特征数量
                feature_names.append(f"Team_{i+1}_{stat}_Feature_{j+1}")
        feature_names.append(f"Team_{i+1}_Count")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建LightGBM数据集
    lgb_train = lgb.Dataset(X_train_scaled, y_train_encoded)
    lgb_test = lgb.Dataset(X_test_scaled, y_test_encoded, reference=lgb_train)
    
    # 设置LightGBM参数 - 增加正则化
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': ['multi_logloss', 'multi_error'],
        'num_leaves': 31,  # 减少叶子数量
        'max_depth': 6,    # 限制树深度
        'learning_rate': 0.01,  # 降低学习率
        'feature_fraction': 0.7,  # 减少特征使用比例
        'bagging_fraction': 0.7,  # 减少数据使用比例
        'bagging_freq': 5,
        'min_child_samples': 20,  # 增加最小叶子样本数
        'min_child_weight': 0.001,
        'lambda_l1': 0.1,  # L1正则化
        'lambda_l2': 0.1,  # L2正则化
        'verbose': -1,
        'seed': 42
    }
    
    # 创建回调函数
    evals_result = {}
    callbacks = [
        lgb.record_evaluation(evals_result),
        lgb.early_stopping(stopping_rounds=30),  # 更严格的早停
        lgb.log_evaluation(period=50)
    ]
    
    # 训练模型
    print("开始训练LightGBM模型...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,  # 减少最大迭代次数
        valid_sets=[lgb_train, lgb_test],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    # 保存模型
    model.save_model('lgbm_model.txt')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("模型已保存")
    
    # 绘制训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(evals_result['train']['multi_logloss'], label='Train')
    plt.plot(evals_result['valid']['multi_logloss'], label='Validation')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('Training and Validation Log Loss')
    plt.legend()
    plt.show()
    
    # 预测
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 计算指标
    test_metrics = calculate_metrics(
        y_test_encoded, y_pred, y_pred_proba, 
        label_encoder, "测试集"
    )
    
    # 绘制特征重要性
    feature_imp = plot_feature_importance(model, feature_names)
    
    # 打印模型总结
    print("\n模型总结:")
    print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    print(f"测试集F1分数: {test_metrics['f1']:.4f}")
    print(f"测试集对数损失: {test_metrics['log_loss']:.4f}")
    print(f"测试集宏观平均AUC: {test_metrics['roc_auc']['macro']:.4f}")
    print(f"测试集微观平均AUC: {test_metrics['roc_auc']['micro']:.4f}")
    
    # 输出最佳迭代次数
    print(f"最佳迭代次数: {model.best_iteration}")
    
    # 对训练集进行评估
    y_train_pred_proba = model.predict(X_train_scaled)
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    
    train_metrics = calculate_metrics(
        y_train_encoded, y_train_pred, y_train_pred_proba, 
        label_encoder, "训练集"
    )
    
    # 计算泛化差距
    generalization_gap = train_metrics['accuracy'] - test_metrics['accuracy']
    print(f"\n泛化差距（训练准确率 - 测试准确率）: {generalization_gap:.4f}")

if __name__ == "__main__":
    main()