import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

warnings.filterwarnings('ignore')

# ==========================================
# 模块一：特征工程提取器 - KMeans 聚类距离特征
# ==========================================
def create_kmeans_features(X_train, X_val, n_clusters=5, random_state=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_train_scaled)
    
    dist_train = kmeans.transform(X_train_scaled)
    dist_val = kmeans.transform(X_val_scaled)
    
    X_train_enhanced = np.hstack([X_train, dist_train])
    X_val_enhanced = np.hstack([X_val, dist_val])
    
    return X_train_enhanced, X_val_enhanced

# ==========================================
# 模块二：构建 SOTA 级别的 Stacking 架构
# ==========================================
def build_stacking_model(xgb_params, lgb_params, cat_params):
    xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss')
    lgb_model = LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
    cat_model = CatBoostClassifier(**cat_params, random_state=42, verbose=0)
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

    estimators = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('rf', rf_model)
    ]

    meta_learner = LogisticRegression(C=0.1, class_weight='balanced')

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=1,  
        passthrough=True 
    )
    return stacking_clf

# ==========================================
# 模块三：Optuna 贝叶斯超参数优化
# ==========================================
def objective(trial, X, y):
    k_clusters = trial.suggest_int('k_clusters', 3, 10)
    
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 300),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True)
    }
    
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 300),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 60)
    }

    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 100, 300),
        'depth': trial.suggest_int('cat_depth', 4, 8),
        'learning_rate': trial.suggest_float('cat_lr', 0.01, 0.1, log=True)
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
    f1_scores = []
    auc_scores = []
    acc_scores = [] 
    
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, y_fold_train = X[train_idx], y[train_idx]
        X_fold_val, y_fold_val = X[val_idx], y[val_idx]
        
        X_fold_train_aug, X_fold_val_aug = create_kmeans_features(X_fold_train, X_fold_val, n_clusters=k_clusters)
        
        model = build_stacking_model(xgb_params, lgb_params, cat_params)
        model.fit(X_fold_train_aug, y_fold_train)
        
        preds = model.predict(X_fold_val_aug)
        
        if len(np.unique(y)) == 2:
            probas = model.predict_proba(X_fold_val_aug)[:, 1]
            auc = roc_auc_score(y_fold_val, probas)
        else:
            probas = model.predict_proba(X_fold_val_aug)
            auc = roc_auc_score(y_fold_val, probas, multi_class='ovr')
            
        f1 = f1_score(y_fold_val, preds, average='macro')
        acc = accuracy_score(y_fold_val, preds)
        
        f1_scores.append(f1)
        auc_scores.append(auc)
        acc_scores.append(acc)
        
    blend_score = 0.3 * np.mean(acc_scores) + 0.4 * np.mean(f1_scores) + 0.3 * np.mean(auc_scores)
    return blend_score

# ==========================================
# 主程序：训练并测试
# ==========================================
if __name__ == "__main__":
    print("正在加载训练集和测试集...")
    try:
        train_df = pd.read_csv(r"C:\Users\huang\Desktop\QG数据集训练\QG_train.csv")
        test_df = pd.read_csv(r"C:\Users\huang\Desktop\QG数据集训练\QG_test.csv")
        print("✓ 数据读取成功！")
    except FileNotFoundError as e:
        print(f"❌ 路径错误：找不到文件。\n详细报错: {e}")
        exit() 

    # 1. 识别目标列名
    target_col = 'label' if 'label' in train_df.columns else 'target'
    
    # 2. 智能筛选特征列（剔除 target 和各种可能的无效 ID 列）
    drop_candidates = [target_col, 'id', 'ID', 'Id', 'Unnamed: 0']
    features = [col for col in train_df.columns if col not in drop_candidates]
    
    # 3. 提取训练集特征和标签
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    
    # 4. 提取测试集 ID 
    test_ids = None
    for col in ['id', 'ID', 'Id']:
        if col in test_df.columns:
            test_ids = test_df[col].values
            break
    if test_ids is None:
        test_ids = range(1, len(test_df) + 1)
        
    # 5. 【关键修复点】：严格强制测试集只使用与训练集相同的特征列！
    # 就算测试集多出了一万个没用的列，也会被过滤掉，保证维度 100% 对齐。
    X_test = test_df[features].values

    print("🚀 启动 Optuna 寻找绝佳参数...")
    study = optuna.create_study(direction="maximize", study_name="Kaggle_Stacking")
    
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=6) 
    
    print("\n🏆 最优混合得分: ", study.best_value)
    best_params = study.best_params
    
    print("\n🔥 正在使用全部数据训练最终模型...")
    X_train_enhanced, X_test_enhanced = create_kmeans_features(X_train, X_test, n_clusters=best_params['k_clusters'])
    
    final_xgb_params = {'n_estimators': best_params['xgb_n_estimators'], 'max_depth': best_params['xgb_max_depth'], 'learning_rate': best_params['xgb_lr']}
    final_lgb_params = {'n_estimators': best_params['lgb_n_estimators'], 'max_depth': best_params['lgb_max_depth'], 'learning_rate': best_params['lgb_lr'], 'num_leaves': best_params['lgb_num_leaves']}
    final_cat_params = {'iterations': best_params['cat_iterations'], 'depth': best_params['cat_depth'], 'learning_rate': best_params['cat_lr']}
    
    final_model = build_stacking_model(final_xgb_params, final_lgb_params, final_cat_params)
    final_model.fit(X_train_enhanced, y_train)
    
    print("🎯 正在对测试集生成预测结果...")
    final_predictions = final_model.predict(X_test_enhanced)
    
    # 组装结果
    submission = pd.DataFrame({
        'id': test_ids,
        'label': final_predictions
    })
    
    output_file = r"C:\Users\huang\Desktop\QG数据集训练\submission.csv"
    submission.to_csv(output_file, index=False)
    
    print(f"\n✅ 预测彻底完成！格式严格对齐要求！")
    print(f"🎉 CSV 提交文件已成功保存到：{output_file}")