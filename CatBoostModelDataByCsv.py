import pandas as pd 
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import pickle

if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file)

    my_df = pd.DataFrame(data)
  
    X = my_df.drop('is_sure', axis=1)
    y = my_df['is_sure']

    # 计算类别权重
    class_counts = y.value_counts()
    total_samples = len(y)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # 定义CatBoost模型
    model = CatBoostClassifier(iterations=1000, 
                               colsample_bylevel=0.5,
                               learning_rate=0.1, 
                               depth=6,
                               l2_leaf_reg=0.1,
                               random_strength=1,
                               class_weights=class_weights)  

    # 使用交叉验证进行模型评估
    scores = cross_val_score(model, X, y, cv=5, scoring='precision')

    # 输出交叉验证结果
    print("Precision:", scores.mean(), "+/-", scores.std())

    # 训练最终模型
    model.fit(X, y)

    # 保存模型
    with open('catboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # 输出分类报告
    y_pred = model.predict(X)
    print(classification_report(y, y_pred, digits=5))
