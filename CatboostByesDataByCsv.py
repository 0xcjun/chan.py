import pandas as pd 
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pickle

if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file)

    my_df = pd.DataFrame(data)
  
    X = my_df.drop('is_sure', axis=1)
    y = my_df['is_sure']

    # 定义类别权重
    class_counts = y.value_counts()
    total_samples = len(y)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # 定义CatBoost模型
    model = CatBoostClassifier(iterations=1000, 
                               colsample_bylevel=0.5,
                               class_weights=class_weights,
                               verbose=0)  

    # 定义超参数搜索空间
    search_spaces = {
        'learning_rate': Real(0.01, 1.0),
        'depth': Integer(1, 10),
        'l2_leaf_reg': Real(0.01, 10.0),
        'random_strength': Real(1, 50)
    }

    # 使用BayesSearchCV进行贝叶斯优化
    opt = BayesSearchCV(model, search_spaces, n_iter=50, cv=5, scoring='precision', n_jobs=-1)

    # 进行参数搜索
    opt.fit(X, y)

    # 输出最佳参数和性能
    print("Best parameters found:", opt.best_params_)
    print("Best precision found:", opt.best_score_)

    # 训练最终模型
    best_model = opt.best_estimator_
    best_model.fit(X, y)

    # 保存模型
    with open('catboost_model_bayes.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # 输出分类报告
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred, digits=5))
