import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file)

    my_df = pd.DataFrame(data)
  
    X = my_df.drop('is_sure', axis=1)
    y = my_df['is_sure']
    # 使用 SimpleImputer 对缺失值进行填充
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    # 定义 MLP 模型
    model = MLPClassifier()

    # 定义超参数搜索范围
    parameters = {
        'hidden_layer_sizes': [(100,), (50,), (100, 50), (50, 25)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }

    # 使用 GridSearchCV 进行超参数搜索
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='precision')
    grid_search.fit(X, y)

    # 输出最佳参数组合
    print("Best parameters found:")
    print(grid_search.best_params_)
    print()

    # 最佳模型
    best_model = grid_search.best_estimator_

    # 训练最佳模型
    best_model.fit(X, y)

    # 保存模型
    with open('mlp_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # 输出分类报告
    y_pred = best_model.predict(X)
    print("Classification report:")
    print(classification_report(y, y_pred, digits=5))
