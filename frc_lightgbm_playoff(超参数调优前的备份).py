import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, roc_curve, auc
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 读取测试数据
data = pd.read_csv('playoffs.csv')
print("payoffs数据读取完成")

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

# 创建验证集 (从训练集中划分)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# 设置LightGBM参数
#params = {
#    'objective': 'multiclass',
#    'num_class': 3,
#    'metric': 'multi_logloss',
#    'boosting_type': 'gbdt',
#    'learning_rate': 0.05,
#    'num_leaves': 31,
#    'max_depth': -1,
#    'min_child_samples': 20,
#    'subsample': 0.8,
#    'colsample_bytree': 0.8,
#    'reg_alpha': 0.1,
#    'reg_lambda': 0.1,
#    'random_state': 42,
#    'n_jobs': -1,
#    'verbosity': -1
#}
"""
过拟合（训练集表现好，验证集差），可以尝试：增加 min_data_in_leaf、增加 lambda_l1 或 lambda_l2、减小 feature_fraction 或 bagging_fraction。
欠拟合（训练集和验证集表现都差），可以尝试：增加 num_leaves、减小 min_data_in_leaf、减小正则化强度。

数据集存在类别不平衡，可以设置 is_unbalance=True 让算法自动调整

进一步尝试不同的 boosting_type（如 gbdt, dart, goss），或者回过头来用更小的学习率（如 0.01, 0.005）配合更大的 n_estimators 进行精细优化
"""

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,  # 降低学习率
    'num_leaves': 20,  # 过拟合减少叶子数量。 小于 2^(max_depth), 31-300	
    'max_depth': 5,  # 限制树深度
    'min_data_in_leaf': 30,  # 过拟合增加叶子最小数据量
    'min_split_gain': 0.01,  # 增加分裂最小增益
    'feature_fraction': 0.7,  # 减少特征采样比例
    'bagging_fraction': 0.7,  # 减少数据采样比例
    'bagging_freq': 5,
    'lambda_l1': 0.1,  # 增加L1正则化
    'lambda_l2': 0.1,  # 增加L2正则化
    'min_child_samples': 20,  # 增加叶子最小样本数
    'verbose': -1,
    'random_state': 42
}

# 使用最新的callback写法:cite[1]:cite[2]:cite[3]
print("开始训练LightGBM模型...")
evals_result = {}

# 定义callback列表
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),  # 早停法:cite[2]
    lgb.log_evaluation(period=100),  # 定期输出评估结果
    lgb.record_evaluation(evals_result)  # 记录评估结果:cite[1]:cite[3]
]

# 训练模型
model = lgb.LGBMClassifier(**params)
model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

print("LightGBM模型训练完成")

# 预测概率
y_train_pred = model.predict_proba(X_train_scaled)
y_test_pred = model.predict_proba(X_test_scaled)

# 计算评估指标
def evaluate(y_true, y_pred):
    y_pred_label = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label, average='weighted')

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 3:  
        print("无平局")
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
    if len(unique_classes) < 3:  
        micro_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro', labels=[0, 1, 2])
    else:
        micro_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')
    
    return accuracy, f1, loss, auc_scores, macro_auc, micro_auc, y_pred_label

train_accuracy, train_f1, train_loss, train_auc, train_macro_auc, train_micro_auc, train_pred_labels = evaluate(y_train, y_train_pred)
test_accuracy, test_f1, test_loss, test_auc, test_macro_auc, test_micro_auc, test_pred_labels = evaluate(y_test, y_test_pred)

print("训练集评估指标:")
print(f"准确率: {train_accuracy:.4f}, F1分数: {train_f1:.4f}, 对数损失: {train_loss:.4f}")
print(f"各类别AUC: {[f'{auc:.4f}' for auc in train_auc]}")
print(f"宏观AUC: {train_macro_auc:.4f}, 微观AUC: {train_micro_auc:.4f}")

print("\n测试集评估指标:")
print(f"准确率: {test_accuracy:.4f}, F1分数: {test_f1:.4f}, 对数损失: {test_loss:.4f}")
print(f"各类别AUC: {[f'{auc:.4f}' for auc in test_auc]}")
print(f"宏观AUC: {test_macro_auc:.4f}, 微观AUC: {test_micro_auc:.4f}")

# 绘制训练过程中的损失曲线:cite[6]
if evals_result:
    plt.figure(figsize=(10, 6))
    for valid_name in evals_result:
        if 'multi_logloss' in evals_result[valid_name]:
            plt.plot(evals_result[valid_name]['multi_logloss'], label=f'{valid_name} loss')
    plt.xlabel('Iterations')
    plt.ylabel('Multi Logloss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制特征重要性:cite[2]
lgb.plot_importance(model, max_num_features=20, figsize=(10, 8))
plt.title('Feature Importance')
plt.show()

# 绘制AUC曲线
def plot_auc_curves(y_true, y_pred, title):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    labels = ['Win', 'Loss', 'Draw']
    
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
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', lw=2, 
             label=f'Macro-average (AUC = {macro_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

plot_auc_curves(y_train, y_train_pred, 'Train ROC Curves')
plot_auc_curves(y_test, y_test_pred, 'Test ROC Curves')

# 输出前20个预测结果和实际结果
print("\n训练集前20个预测结果:")
for i in range(min(20, len(y_train))):
    pred_label = train_pred_labels[i]
    true_label = y_train[i]
    print(f"样本 {i+1}: 预测={pred_label}({'Win' if pred_label==0 else 'Loss' if pred_label==1 else 'Draw'}), "
          f"实际={true_label}({'Win' if true_label==0 else 'Loss' if true_label==1 else 'Draw'})")

print("\n测试集前20个预测结果:")
for i in range(min(20, len(y_test))):
    pred_label = test_pred_labels[i]
    true_label = y_test[i]
    print(f"样本 {i+1}: 预测={pred_label}({'Win' if pred_label==0 else 'Loss' if pred_label==1 else 'Draw'}), "
          f"实际={true_label}({'Win' if true_label==0 else 'Loss' if true_label==1 else 'Draw'})")

# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred_label, title):
    cm = confusion_matrix(y_true, y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Win', 'Loss', 'Draw'])
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_train, train_pred_labels, 'Train Confusion Matrix')
plot_confusion_matrix(y_test, test_pred_labels, 'Test Confusion Matrix')

# 输出最佳迭代次数和其他训练信息
if hasattr(model, 'best_iteration_'):
    print(f"\n最佳迭代次数: {model.best_iteration_}")
    
if hasattr(model, 'best_score_'):
    print(f"最佳得分: {model.best_score_}")