import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.experimental import enable_hist_gradient_boosting  # enable this experimental module
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle

if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file)

    my_df = pd.DataFrame(data)
  
    X = my_df.drop('is_sure', axis=1)
    y = my_df['is_sure']

    # 定义 HistGradientBoostingClassifier 模型
    model = HistGradientBoostingClassifier()

    # 训练模型
    model.fit(X, y)

    # 保存模型
    with open('hist_gradient_boosting_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # 输出分类报告
    y_pred = model.predict(X)
    print(classification_report(y, y_pred, digits=5))
